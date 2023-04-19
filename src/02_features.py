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
