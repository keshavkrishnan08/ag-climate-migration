"""Phase 2: Feature engineering — yields + climate + switching.

Builds the complete feature matrix for all county-crop-year observations.
Works with actual downloaded data:
    - NASS yields: data/raw/nass/nass_county_yields.parquet
    - Climate: data/raw/prism/county_climate_annual.parquet (NOAA nClimDiv, °F)
    - ACS demographics: data/raw/census/acs_county_demographics.parquet
    - Farm operations: data/raw/nass/nass_farm_operations.parquet
    - ERS Atlas: data/raw/other/ers_atlas/*.parquet
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

F_TO_C = lambda f: (f - 32) * 5 / 9


# ---------------------------------------------------------------------------
# Core derived variables
# ---------------------------------------------------------------------------
def compute_gdd_from_monthly(
    tmax_f: float,
    tmin_f: float,
    base_c: float,
    upper_c: float,
    days_in_month: int = 30
) -> float:
    """Approximate monthly GDD from monthly avg tmax/tmin (given in °F).

    Args:
        tmax_f: Monthly average max temperature in °F.
        tmin_f: Monthly average min temperature in °F.
        base_c: Crop base temperature in °C.
        upper_c: Crop upper threshold in °C.
        days_in_month: Days in the month.

    Returns:
        Approximate GDD for that month.
    """
    if np.isnan(tmax_f) or np.isnan(tmin_f):
        return np.nan
    tmax_c = F_TO_C(tmax_f)
    tmin_c = F_TO_C(tmin_f)
    tavg = (tmax_c + tmin_c) / 2.0
    effective = min(tavg, upper_c)
    daily_gdd = max(0.0, effective - base_c)
    return daily_gdd * days_in_month


def compute_yield_anomaly(yields_series: pd.Series) -> pd.Series:
    """Remove technology trend from yield to isolate climate signal.

    Fits linear + quadratic trend. Returns z-score residuals.

    Args:
        yields_series: County-crop yields indexed by year.

    Returns:
        Detrended yield anomaly series (z-score).
    """
    s = yields_series.dropna()
    if len(s) < 10:
        return pd.Series(np.nan, index=yields_series.index)

    years = s.index.values.astype(float)
    values = s.values.astype(float)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        coeffs = np.polyfit(years, values, deg=2)
    trend = np.polyval(coeffs, years)
    residuals = values - trend

    std = residuals.std()
    anomaly = residuals / std if std > 0 else residuals
    return pd.Series(anomaly, index=s.index)


# ---------------------------------------------------------------------------
# Load real data files
# ---------------------------------------------------------------------------
def load_nass_yields() -> pd.DataFrame:
    """Load NASS county yields, filter to 1950+ with valid yields.

    Deduplicates to one record per county-crop-year (NASS bulk has
    multiple records per group for different practices/coverage types).

    Returns:
        DataFrame: fips, year, crop, yield_bu_acre, acres_harvested.
    """
    path = DATA_RAW / 'nass' / 'nass_county_yields.parquet'
    df = pd.read_parquet(path)
    df = df[df['year'] >= 1950].copy()
    df = df.dropna(subset=['yield_bu_acre'])
    df['fips'] = df['fips'].astype(str).str.zfill(5)
    # Remove "other counties" aggregates
    df = df[~df['fips'].str.endswith(('998', '999'))]

    # Deduplicate: one record per county-crop-year
    n_before = len(df)
    df = df.groupby(['fips', 'year', 'crop']).agg({
        'yield_bu_acre': 'first',
        'acres_harvested': 'first',
    }).reset_index()

    logger.info(f"NASS yields: {len(df):,} rows (deduped from {n_before:,}), "
                f"{df['fips'].nunique()} counties, {df['year'].min()}-{df['year'].max()}")
    return df


def load_climate() -> pd.DataFrame:
    """Load NOAA nClimDiv county climate data (already parsed to parquet).

    Data is in °F. Growing season aggregates already computed.

    Returns:
        DataFrame: fips, year, tmax_growing_avg, tmin_growing_avg,
        precip_growing_total, tmax_july, pdsi_growing_avg, cdd_annual.
    """
    path = DATA_RAW / 'prism' / 'county_climate_annual.parquet'
    df = pd.read_parquet(path)
    df['fips'] = df['fips'].astype(str).str.zfill(5)
    logger.info(f"Climate: {len(df):,} county-years, {df['fips'].nunique()} counties")
    return df


def load_monthly_climate() -> pd.DataFrame:
    """Load full monthly climate for GDD computation and trend analysis.

    Returns:
        DataFrame with monthly tmax/tmin/precip/pdsi columns per county-year.
    """
    path = DATA_RAW / 'prism' / 'county_climate_monthly.parquet'
    df = pd.read_parquet(path)
    df['fips'] = df['fips'].astype(str).str.zfill(5)
    return df


def load_acs_demographics() -> pd.DataFrame:
    """Load Census ACS county demographics.

    Returns:
        DataFrame: fips, year, total_population, median_household_income, etc.
    """
    path = DATA_RAW / 'census' / 'acs_county_demographics.parquet'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df['fips'] = df['fips'].astype(str).str.zfill(5)
    for col in ['total_population', 'median_household_income', 'poverty_count', 'median_home_value']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_ers_atlas() -> pd.DataFrame:
    """Load ERS Atlas county classifications.

    Returns:
        DataFrame with county typology codes and economic indicators.
    """
    path = DATA_RAW / 'other' / 'ers_atlas' / 'People.csv'
    if not path.exists():
        return pd.DataFrame()
    # ERS is in long format: FIPS, State, County, Attribute, Value
    df = pd.read_csv(path, dtype=str, low_memory=False, encoding='latin-1')
    if 'FIPS' in df.columns:
        df['fips'] = df['FIPS'].str.zfill(5)
    return df


# ---------------------------------------------------------------------------
# Feature builders (vectorized for real data scale)
# ---------------------------------------------------------------------------
def build_climate_features(climate_annual: pd.DataFrame, climate_monthly: pd.DataFrame) -> pd.DataFrame:
    """Build all climate feature blocks from real nClimDiv data.

    Produces per county-year:
        - Current season: tmax_july_c, tmin_growing_c, precip_growing, pdsi, cdd
        - GDD: corn_gdd, soy_gdd, wheat_gdd, cotton_gdd (from monthly data)
        - Trends: 10-year slopes for tmax, precip
        - Anomalies: z-scores vs 1981-2010 baseline

    Args:
        climate_annual: Annual growing-season aggregates.
        climate_monthly: Monthly data for GDD computation.

    Returns:
        DataFrame with climate features per county-year.
    """
    logger.info("Building climate features...")

    # --- Current season features (convert °F to °C) ---
    cf = climate_annual[['fips', 'year']].copy()
    cf['tmax_july_c'] = F_TO_C(climate_annual['tmax_july'])
    cf['tmax_growing_c'] = F_TO_C(climate_annual['tmax_growing_avg'])
    cf['tmin_growing_c'] = F_TO_C(climate_annual['tmin_growing_avg'])
    cf['precip_growing'] = climate_annual['precip_growing_total']
    cf['pdsi_growing'] = climate_annual['pdsi_growing_avg']
    cf['cdd_annual'] = climate_annual['cdd_annual']

    # --- GDD from monthly data (vectorized) ---
    logger.info("  Computing crop-specific GDD from monthly temps (vectorized)...")
    gdd_thresholds = CONFIG['gdd_thresholds']
    days_per_month = {4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30}
    growing_months = list(range(4, 10))

    for crop_key, thresholds in gdd_thresholds.items():
        base_c = thresholds['base']
        upper_c = thresholds['upper']

        total_gdd = np.zeros(len(climate_monthly))
        for m in growing_months:
            tmax_c = F_TO_C(climate_monthly[f'tmax_m{m:02d}'].values)
            tmin_c = F_TO_C(climate_monthly[f'tmin_m{m:02d}'].values)
            tavg = (tmax_c + tmin_c) / 2.0
            effective = np.minimum(tavg, upper_c)
            daily_gdd = np.maximum(0.0, effective - base_c)
            total_gdd += daily_gdd * days_per_month[m]

        gdd_df = climate_monthly[['fips', 'year']].copy()
        gdd_df[f'gdd_{crop_key}'] = total_gdd
        cf = cf.merge(gdd_df, on=['fips', 'year'], how='left')

    # --- 10-year trends (vectorized with groupby + rolling) ---
    logger.info("  Computing 10-year climate trends...")
    cf = cf.sort_values(['fips', 'year'])

    for var in ['tmax_july_c', 'precip_growing', 'cdd_annual']:
        slope_col = f'{var}_trend10'
        slopes = []
        for fips, group in cf.groupby('fips'):
            group = group.sort_values('year')
            vals = group[var].values
            yrs = group['year'].values.astype(float)
            s = np.full(len(vals), np.nan)
            for i in range(10, len(vals)):
                window_y = yrs[i - 10:i]
                window_v = vals[i - 10:i]
                mask = ~np.isnan(window_v)
                if mask.sum() >= 5:
                    sl, _, _, _, _ = stats.linregress(window_y[mask], window_v[mask])
                    s[i] = sl
            slopes.extend(s)
        cf[slope_col] = slopes

    # --- Anomalies vs 1981-2010 baseline ---
    logger.info("  Computing climate anomalies vs 1981-2010 baseline...")
    baseline = cf[(cf['year'] >= 1981) & (cf['year'] <= 2010)]

    for var in ['tmax_july_c', 'precip_growing', 'pdsi_growing']:
        anom_col = f'{var}_anomaly'
        bl_stats = baseline.groupby('fips')[var].agg(['mean', 'std']).rename(
            columns={'mean': f'{var}_bl_mean', 'std': f'{var}_bl_std'}
        )
        cf = cf.merge(bl_stats, left_on='fips', right_index=True, how='left')
        cf[anom_col] = (cf[var] - cf[f'{var}_bl_mean']) / cf[f'{var}_bl_std'].replace(0, np.nan)
        cf = cf.drop(columns=[f'{var}_bl_mean', f'{var}_bl_std'])

    # Count extreme heat months (tmax > 95°F = 35°C) — vectorized
    heat_count = np.zeros(len(climate_monthly))
    for m in growing_months:
        tmax_col = f'tmax_m{m:02d}'
        heat_count += (climate_monthly[tmax_col].values > 95).astype(int)
    heat_df = climate_monthly[['fips', 'year']].copy()
    heat_df['extreme_heat_months'] = heat_count
    cf = cf.merge(heat_df, on=['fips', 'year'], how='left')

    logger.info(f"  Climate features: {cf.shape[1] - 2} variables for {len(cf):,} county-years")
    return cf


def build_technology_features(yields_df: pd.DataFrame) -> pd.DataFrame:
    """Build technology trend features using vectorized rolling regression.

    For each county-crop, computes 15-year rolling yield trend slope.

    Args:
        yields_df: NASS yields with fips, year, crop, yield_bu_acre.

    Returns:
        DataFrame with yield_trend_slope_15yr per county-crop-year.
    """
    logger.info("Building technology trend features...")

    results = []
    for (fips, crop), group in yields_df.groupby(['fips', 'crop']):
        group = group.sort_values('year')
        yrs = group['year'].values.astype(float)
        yld = group['yield_bu_acre'].values.astype(float)

        slopes = np.full(len(yrs), np.nan)
        intercepts = np.full(len(yrs), np.nan)

        for i in range(len(yrs)):
            mask = (yrs >= yrs[i] - 15) & (yrs <= yrs[i]) & ~np.isnan(yld)
            if mask.sum() >= 5:
                sl, ic, _, _, _ = stats.linregress(yrs[mask], yld[mask])
                slopes[i] = sl
                intercepts[i] = ic

        df_out = group[['fips', 'year', 'crop']].copy()
        df_out['yield_trend_slope_15yr'] = slopes
        df_out['yield_trend_intercept'] = intercepts
        results.append(df_out)

    result = pd.concat(results, ignore_index=True)
    logger.info(f"  Tech features: {len(result):,} county-crop-years")
    return result


def build_switching_proxy(yields_df: pd.DataFrame) -> pd.DataFrame:
    """Build crop switching proxy from NASS acreage share changes.

    Without CDL, we approximate switching as change in acreage shares
    at county level between consecutive years.

    Args:
        yields_df: NASS yields with acres_harvested.

    Returns:
        DataFrame with switching proxy features per county-year.
    """
    logger.info("Building switching proxy from NASS acreage shares...")

    if 'acres_harvested' not in yields_df.columns or yields_df['acres_harvested'].isna().all():
        logger.warning("No acreage data — skipping switching proxy")
        return pd.DataFrame()

    # Compute county-level acreage shares
    county_year_total = yields_df.groupby(['fips', 'year'])['acres_harvested'].sum().reset_index()
    county_year_total.columns = ['fips', 'year', 'total_acres']

    merged = yields_df.merge(county_year_total, on=['fips', 'year'])
    merged['acreage_share'] = merged['acres_harvested'] / merged['total_acres'].replace(0, np.nan)

    # For each county-year, compute total share change (proxy for switching)
    results = []
    for fips, county_data in merged.groupby('fips'):
        pivot = county_data.pivot_table(index='year', columns='crop', values='acreage_share', aggfunc='first')
        pivot = pivot.sort_index()

        # Year-over-year absolute change in shares
        share_change = pivot.diff().abs()
        annual_switching = share_change.sum(axis=1) / 2  # divide by 2 since changes sum to ~0

        # 5-year rolling switching rate
        rolling_5yr = annual_switching.rolling(5, min_periods=3).mean()

        # Switching velocity (acceleration)
        for year in pivot.index:
            record = {
                'fips': fips,
                'year': year,
                'switching_rate_proxy': annual_switching.get(year, np.nan),
                'switching_rate_5yr': rolling_5yr.get(year, np.nan),
            }
            results.append(record)

    result = pd.DataFrame(results)
    logger.info(f"  Switching proxy: {len(result):,} county-years")
    return result


def build_demographic_features(acs: pd.DataFrame) -> pd.DataFrame:
    """Build demographic features from ACS data.

    Args:
        acs: ACS demographics with population, income, etc.

    Returns:
        DataFrame with county-year demographic features.
    """
    if acs.empty:
        return pd.DataFrame()

    logger.info("Building demographic features from ACS...")
    df = acs[['fips', 'year']].copy()

    if 'total_population' in acs.columns:
        df['log_population'] = np.log1p(acs['total_population'].fillna(0))
    if 'median_household_income' in acs.columns:
        df['log_median_income'] = np.log1p(acs['median_household_income'].fillna(0))
    if 'poverty_count' in acs.columns and 'total_population' in acs.columns:
        df['poverty_rate'] = acs['poverty_count'] / acs['total_population'].replace(0, np.nan)

    logger.info(f"  Demographic features: {df.shape[1] - 2} variables")
    return df


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------
def build_feature_matrix() -> pd.DataFrame:
    """Build complete feature matrix from real downloaded data.

    Returns:
        Complete panel dataset ready for LightGBM modeling.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("=" * 60)

    # --- Load data ---
    yields_df = load_nass_yields()
    climate_annual = load_climate()
    climate_monthly = load_monthly_climate()
    acs = load_acs_demographics()

    # --- Start with yield panel ---
    panel = yields_df[['fips', 'year', 'crop', 'yield_bu_acre', 'acres_harvested']].copy()
    panel = panel.dropna(subset=['yield_bu_acre'])
    logger.info(f"Base panel: {len(panel):,} rows")

