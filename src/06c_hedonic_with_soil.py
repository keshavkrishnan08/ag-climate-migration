"""Phase 5C: Hedonic farmland valuation with soil productivity controls.

Reviewer concern: the baseline hedonic (06b) uses climate + socioeconomic
controls but no soil quality control. Omitting soil quality may bias the
climate coefficient if soil quality correlates with both temperature and land
values (e.g., rich soils in the Corn Belt → high values AND moderate temps).

This script:
1. Constructs a county-level NCCPI proxy (soil productivity index) from
   maximum observed corn yield 1950–2023, normalized to [0,1]. Max yield
   captures soil potential under optimal weather rather than average climate
   response (Schlenker & Roberts 2009 precedent).
2. Adds an ERS HiAmenity binary control from the USDA ERS Rural Atlas 2024,
   directly addressing the amenity channel.
3. Re-estimates the hedonic regression with both new controls.
4. Compares: climate coefficients, R², and stranded asset estimate
   with vs without soil controls.

Model (baseline, 06b):
    log(V) = β₀ + β_T·T + β_T²·T² + β_P·P + β_pop·log(pop)
           + β_inc·log(inc) + state_FE + ε

Model (soil-controlled, this script):
    log(V) = β₀ + β_T·T + β_T²·T² + β_P·P + β_pop·log(pop)
           + β_inc·log(inc) + β_soil·nccpi_proxy
           + β_amenity·hi_amenity + state_FE + ε

Output:
    results/stranded_assets/hedonic_soil_stranded_2050.parquet
    results/stranded_assets/hedonic_soil_comparison.json
    state/headline_numbers_preliminary.json  (soil_controlled fields added)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
PROJECTIONS_DIR = PROJECT_ROOT / 'data' / 'projections'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Growing season months (April–September)
GROWING_MONTHS = [4, 5, 6, 7, 8, 9]

# CPI deflator — 2023 USD
CPI_2022 = 296.8
CPI_2023 = 304.7
DEFLATOR_2022 = CPI_2023 / CPI_2022

# Winsorize bounds
LAND_VALUE_UPPER_PCTILE = 99
LAND_VALUE_LOWER_PCTILE = 1

# USDA Census of Agriculture 2022: total acres in farms by state FIPS.
USDA_STATE_FARM_ACRES_2022 = {
    '01': 8_700_000, '04': 26_000_000, '05': 14_500_000, '06': 25_300_000,
    '08': 31_000_000, '09': 400_000,   '10': 500_000,   '12': 9_700_000,
    '13': 9_600_000, '16': 11_700_000, '17': 26_900_000, '18': 14_700_000,
    '19': 30_600_000, '20': 45_700_000, '21': 13_800_000, '22': 7_700_000,
    '23': 1_300_000, '24': 2_100_000,  '25': 500_000,   '26': 10_000_000,
    '27': 25_700_000, '28': 10_800_000, '29': 28_200_000, '30': 58_100_000,
    '31': 44_500_000, '33': 400_000,   '34': 700_000,   '35': 44_700_000,
    '36': 6_900_000, '37': 8_500_000,  '38': 38_800_000, '39': 13_800_000,
    '40': 33_800_000, '41': 16_400_000, '42': 7_300_000, '44': 70_000,
    '45': 4_700_000, '46': 43_200_000, '47': 10_900_000, '48': 127_000_000,
    '49': 11_000_000, '50': 1_300_000, '51': 7_900_000,  '53': 15_100_000,
    '54': 3_600_000, '55': 14_200_000, '56': 29_500_000,
}


def build_nccpi_proxy(nass_yields: pd.DataFrame) -> pd.DataFrame:
    """Build county-level soil productivity index from peak corn yields.

    NCCPI proxy logic: the all-time maximum corn yield observed in a county
    (1950–2023) reflects soil potential under optimal weather conditions. Rich
    soils reach higher ceilings under good weather. Thin soils plateau lower.
    Normalizing to [0,1] gives a scale-free productivity index comparable to
    the USDA NCCPI (0=worst, 1=best).

    Counties without any corn history receive NaN (handled downstream).

    Args:
        nass_yields: NASS county yields with columns [fips, year, crop,
                     yield_bu_acre].

    Returns:
        DataFrame with columns [fips, max_corn_yield, nccpi_proxy].
    """
    logger.info("Building NCCPI proxy from peak corn yields (1950–2023)...")

    corn = nass_yields[nass_yields['crop'] == 'corn'].copy()
    corn = corn[(corn['yield_bu_acre'] > 0) & corn['yield_bu_acre'].notna()]

    max_yield = corn.groupby('fips')['yield_bu_acre'].max().reset_index()
    max_yield.columns = ['fips', 'max_corn_yield']

    # Normalize to [0, 1]
    y_max = max_yield['max_corn_yield'].max()
    max_yield['nccpi_proxy'] = max_yield['max_corn_yield'] / y_max

    logger.info(
        f"  NCCPI proxy: {len(max_yield)} counties, "
        f"max yield = {y_max:.0f} bu/ac, "
        f"mean proxy = {max_yield['nccpi_proxy'].mean():.3f}"
    )
    return max_yield[['fips', 'nccpi_proxy']]


def build_amenity_control(ers_path: Path) -> pd.DataFrame:
    """Extract HiAmenity binary from ERS Rural Atlas county classifications.

    The USDA ERS Natural Amenities measure (embedded as HiAmenity in the
    Rural Atlas) captures climate amenity value: mild winters, low humidity,
    topographic relief, water access. Including it controls for the amenity
    channel in land values that is correlated with but distinct from
    agricultural climate productivity.

    Args:
        ers_path: Path to ERS atlas CountyClassifications.csv.

    Returns:
        DataFrame with columns [fips, hi_amenity] (int 0/1).
    """
    logger.info("Loading ERS HiAmenity control from Rural Atlas...")

    df = pd.read_csv(ers_path, encoding='latin1')
    # First column is FIPS (unnamed in the CSV)
    fips_col = df.columns[0]
    df = df.rename(columns={fips_col: 'fips_raw'})

    amenity = df[df['Attribute'] == 'HiAmenity'][['fips_raw', 'Value']].copy()
    amenity['fips'] = amenity['fips_raw'].astype(str).str.zfill(5)
    amenity = amenity.rename(columns={'Value': 'hi_amenity'})
    amenity['hi_amenity'] = amenity['hi_amenity'].astype(int)

    logger.info(
        f"  HiAmenity: {len(amenity)} counties, "
        f"{amenity['hi_amenity'].sum()} high-amenity (={amenity['hi_amenity'].mean():.1%})"
    )
    return amenity[['fips', 'hi_amenity']]


def build_cross_section_with_soil(
    land_values: pd.DataFrame,
    climate_monthly: pd.DataFrame,
    acs: pd.DataFrame,
    nass_yields: pd.DataFrame,
    nccpi_proxy: pd.DataFrame,
    amenity: pd.DataFrame,
) -> pd.DataFrame:
    """Build county cross-section with soil and amenity controls.

    Extends the baseline 06b cross-section by merging in:
    - nccpi_proxy (soil productivity index, 0–1)
    - hi_amenity (ERS binary amenity indicator)

    Counties lacking corn history get nccpi_proxy = median (imputed).
    Counties lacking amenity data get hi_amenity = 0 (conservative).

    Args:
        land_values: NASS land values (fips, year, land_value_per_acre).
        climate_monthly: PRISM monthly climate data.
        acs: ACS demographics (fips, year, total_population,
             median_household_income).
        nass_yields: NASS county yields (fips, year, acres_harvested).
        nccpi_proxy: Soil proxy from build_nccpi_proxy.
        amenity: Amenity binary from build_amenity_control.

    Returns:
        DataFrame with one row per county, including nccpi_proxy and
        hi_amenity columns alongside baseline hedonic variables.
    """
    logger.info("Building soil-controlled cross-section...")

    # --- Land value: average 2017 and 2022 ---
    lv_recent = land_values[land_values['year'].isin([2017, 2022])].copy()
    lv_recent.loc[lv_recent['year'] == 2022, 'land_value_per_acre'] *= DEFLATOR_2022
    lv_cs = (
        lv_recent.groupby('fips')['land_value_per_acre']
        .mean()
        .reset_index()
    )

    lo = np.percentile(lv_cs['land_value_per_acre'], LAND_VALUE_LOWER_PCTILE)
    hi = np.percentile(lv_cs['land_value_per_acre'], LAND_VALUE_UPPER_PCTILE)
    lv_cs = lv_cs[
        (lv_cs['land_value_per_acre'] >= lo) &
        (lv_cs['land_value_per_acre'] <= hi)
    ].copy()
    logger.info(f"  Land values after winsorize: {len(lv_cs)} counties")

    # --- Climate: 2019-2023 average ---
    clim_window = climate_monthly[climate_monthly['year'].between(2019, 2023)].copy()
    precip_cols = [f'precip_m{m:02d}' for m in GROWING_MONTHS]
    clim_window['precip_growing'] = clim_window[precip_cols].sum(axis=1)
    clim_window['tmax_july'] = clim_window['tmax_m07']
    clim_cs = (
        clim_window.groupby('fips')[['tmax_july', 'precip_growing']]
        .mean()
        .reset_index()
    )

    # --- ACS controls: 2019-2023 average ---
    acs_window = acs[acs['year'].between(2019, 2023)].copy()
    acs_cs = (
        acs_window.groupby('fips')[['total_population', 'median_household_income']]
        .mean()
        .reset_index()
    )

    # --- Farm acres: calibrated (identical method to 06b) ---
    nass_recent = nass_yields[nass_yields['year'].between(2017, 2022)].copy()
    max_by_county_year = (
        nass_recent.groupby(['fips', 'year'])['acres_harvested'].max()
    )
    max_acres_df = (
        max_by_county_year.groupby('fips').mean().reset_index()
        .rename(columns={'acres_harvested': 'max_crop_acres'})
    )
    max_acres_df['state'] = max_acres_df['fips'].str[:2]
    state_max_totals = max_acres_df.groupby('state')['max_crop_acres'].sum()
    calib_factors = {}
    for st, usda_acres in USDA_STATE_FARM_ACRES_2022.items():
        our_max = state_max_totals.get(st, 0)
        calib_factors[st] = usda_acres / our_max if our_max > 0 else 5.0
    max_acres_df['calib_factor'] = max_acres_df['state'].map(calib_factors).fillna(5.0)
    max_acres_df['farm_acres'] = max_acres_df['max_crop_acres'] * max_acres_df['calib_factor']
    farm_acres = max_acres_df[['fips', 'farm_acres']]

    # --- Merge baseline ---
    df = lv_cs.merge(clim_cs, on='fips', how='inner')
    df = df.merge(acs_cs, on='fips', how='inner')
    df = df.merge(farm_acres, on='fips', how='left')

    # --- Merge soil + amenity ---
    df = df.merge(nccpi_proxy, on='fips', how='left')
    df = df.merge(amenity, on='fips', how='left')

    # Impute missing nccpi_proxy with median (non-corn counties)
    median_nccpi = df['nccpi_proxy'].median()
    n_imputed = df['nccpi_proxy'].isna().sum()
    df['nccpi_proxy'] = df['nccpi_proxy'].fillna(median_nccpi)
    df['nccpi_imputed'] = (df['nccpi_proxy'] == median_nccpi).astype(int)
    if n_imputed > 0:
        logger.info(f"  NCCPI proxy: imputed {n_imputed} counties with median={median_nccpi:.3f}")

    # Missing amenity → 0 (most non-amenity counties are ag counties)
    df['hi_amenity'] = df['hi_amenity'].fillna(0).astype(int)

    # --- Derived variables ---
    df['tmax_july_sq'] = df['tmax_july'] ** 2
    df['log_land_value'] = np.log(df['land_value_per_acre'])
    df['log_pop'] = np.log(df['total_population'].clip(lower=1))
    df['log_income'] = np.log(df['median_household_income'].clip(lower=1))
    df['state_fips'] = df['fips'].str[:2]

    df = df.dropna(subset=[
        'log_land_value', 'tmax_july', 'precip_growing',
        'log_pop', 'log_income', 'nccpi_proxy',
    ])
    df = df[df['total_population'] > 0]
    df = df[df['median_household_income'] > 0]
    df = df[df['tmax_july'] > 30]
    df = df[df['precip_growing'] >= 0]

    logger.info(f"  Final cross-section (soil-controlled): {len(df)} counties")
    logger.info(f"  nccpi_proxy: mean={df['nccpi_proxy'].mean():.3f}, sd={df['nccpi_proxy'].std():.3f}")
    logger.info(f"  hi_amenity: {df['hi_amenity'].sum()} counties ({df['hi_amenity'].mean():.1%})")

    return df


def estimate_hedonic_with_soil(df: pd.DataFrame) -> tuple:
    """Estimate hedonic regression with NCCPI soil proxy and amenity controls.

    Model:
        log(V) ~ tmax_july + tmax_july_sq + precip_growing
               + log_pop + log_income
               + nccpi_proxy + hi_amenity
               + C(state_fips)

    Uses HC3 heteroskedasticity-consistent standard errors.

    Args:
        df: Cross-section from build_cross_section_with_soil.

    Returns:
        Tuple of (OLS result, DataFrame with fitted values/residuals).
    """
    logger.info("Estimating soil-controlled hedonic regression...")

    formula = (
        "log_land_value ~ tmax_july + tmax_july_sq + precip_growing "
        "+ log_pop + log_income "
        "+ nccpi_proxy + hi_amenity "
        "+ C(state_fips)"
    )
    model = smf.ols(formula=formula, data=df)
    result = model.fit(cov_type='HC3')

    logger.info(f"  N = {int(result.nobs)}")
    logger.info(f"  R² = {result.rsquared:.4f}")
