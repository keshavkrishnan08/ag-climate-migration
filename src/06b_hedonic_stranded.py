"""Phase 5B: Mendelsohn-Nordhaus-Schlenker hedonic farmland valuation.

Cross-sectional hedonic regression of observed farmland values on climate
variables (Mendelsohn, Nordhaus & Shaw 1994; Schlenker, Hanemann & Fisher
2005, 2006). No discount rate required. Captures ALL value channels —
crop yields, amenity value, water, livestock, specialty crops — via
market-revealed land prices.

Model:
    log(land_value) = β₀ + β₁·tmax_july + β₂·tmax_july² + β₃·precip_growing
                    + β₄·log(pop) + β₅·log(income) + state_FE + ε

Stranded value per county:
    delta_value = predicted(current) - predicted(projected)  [$/acre]
    total = delta_value × farm_acres

Aggregate nationally to get hedonic stranded estimate.

Output:
    results/stranded_assets/hedonic_stranded.parquet
    state/headline_numbers_preliminary.json  (hedonic_stranded_B field added)
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

# Growing season months (April–September) for precipitation sum
GROWING_MONTHS = [4, 5, 6, 7, 8, 9]

# CPI deflator — 2023 USD (from config; CPI_2022=296.8, CPI_2023=304.7)
CPI_2022 = 296.8
CPI_2023 = 304.7
DEFLATOR_2022 = CPI_2023 / CPI_2022  # inflate 2022 values to 2023 USD

# Winsorize outliers to avoid leverage from extreme markets (urban fringe)
LAND_VALUE_UPPER_PCTILE = 99
LAND_VALUE_LOWER_PCTILE = 1

# USDA Census of Agriculture 2022: total acres in farms by state FIPS.
# Source: USDA NASS 2022 Census of Agriculture, Table 1.
# Used to calibrate county farm acres derived from NASS cropland data,
# which only covers 8 field crops and substantially undercounts total farmland
# (pasture, rangeland, orchards, fallow, farmsteads, etc.).
# Calibration: state_factor = USDA_total / sum(max_crop_acres_per_county)
# Applied county-proportionally so county shares within each state are preserved.
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


def build_cross_section(
    land_values: pd.DataFrame,
    climate_monthly: pd.DataFrame,
    acs: pd.DataFrame,
    nass_yields: pd.DataFrame,
) -> pd.DataFrame:
    """Build county-level cross-section for hedonic regression.

    Uses:
      - Land value: average of 2017 and 2022 Census of Ag (most recent bracket
        around the 2019-2023 climate window), deflated to 2023 USD.
      - Climate: 2019-2023 average tmax_july (July mean Tmax, °F) and
        precip_growing (Apr-Sep total precip, inches/yr).
      - Controls: 2019-2023 average log(population) and log(median_income).
      - State fixed effects from FIPS prefix.

    Args:
        land_values: NASS land values (fips, year, land_value_per_acre).
        climate_monthly: PRISM monthly (fips, year, tmax_m01..m12,
                         precip_m01..m12). Tmax in °F.
        acs: ACS demographics (fips, year, total_population,
             median_household_income).
        nass_yields: NASS county yields (fips, year, acres_harvested).

    Returns:
        DataFrame with one row per county: fips, state_fips, log_land_value,
        tmax_july, tmax_july_sq, precip_growing, log_pop, log_income,
        farm_acres, land_value_per_acre.
    """
    logger.info("Building hedonic cross-section...")

    # --- Land value: average 2017 and 2022 ---
    lv_recent = land_values[land_values['year'].isin([2017, 2022])].copy()
    # Inflate 2022 values to 2023 USD; 2017 values use 2022 CPI ratio as approx
    lv_recent.loc[lv_recent['year'] == 2022, 'land_value_per_acre'] *= DEFLATOR_2022
    lv_cs = (
        lv_recent.groupby('fips')['land_value_per_acre']
        .mean()
        .reset_index()
        .rename(columns={'land_value_per_acre': 'land_value_per_acre'})
    )
    logger.info(f"  Land values: {len(lv_cs)} counties (2017/2022 avg, 2023 USD)")

    # Winsorize extreme land values (urban fringe parcels distort regression)
    lo = np.percentile(lv_cs['land_value_per_acre'], LAND_VALUE_LOWER_PCTILE)
    hi = np.percentile(lv_cs['land_value_per_acre'], LAND_VALUE_UPPER_PCTILE)
    lv_cs = lv_cs[
        (lv_cs['land_value_per_acre'] >= lo) &
        (lv_cs['land_value_per_acre'] <= hi)
    ].copy()
    logger.info(f"  After winsorize [{lo:.0f}, {hi:.0f}]: {len(lv_cs)} counties")

    # --- Climate: 2019-2023 average ---
    clim_window = climate_monthly[climate_monthly['year'].between(2019, 2023)].copy()
    precip_cols = [f'precip_m{m:02d}' for m in GROWING_MONTHS]
    clim_window['precip_growing'] = clim_window[precip_cols].sum(axis=1)
    clim_window['tmax_july'] = clim_window['tmax_m07']  # July mean Tmax, °F

    clim_cs = (
        clim_window.groupby('fips')[['tmax_july', 'precip_growing']]
        .mean()
        .reset_index()
    )
    logger.info(f"  Climate: {len(clim_cs)} counties (2019-2023 avg)")

    # --- ACS controls: 2019-2023 average ---
    acs_window = acs[acs['year'].between(2019, 2023)].copy()
    acs_cs = (
        acs_window.groupby('fips')[['total_population', 'median_household_income']]
        .mean()
        .reset_index()
    )
    logger.info(f"  ACS demographics: {len(acs_cs)} counties (2019-2023 avg)")

    # --- Farm acres: calibrated to USDA Census of Agriculture 2022 state totals ---
    # Step 1: max single-crop harvested acres per county-year, averaged 2017-2022.
    # Taking max (not sum) avoids multi-crop double-counting (a corn/soy field
    # would appear in both crop rows). Max gives dominant crop acres.
    # Step 2: apply state-specific calibration factor = USDA_total / sum(max_acres).
    # This preserves within-state county proportions while anchoring to USDA totals.
    # Result: county-level farm acres covering ALL land uses (cropland + rangeland
    # + orchards + fallow + farmsteads), consistent with hedonic model scope.
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
