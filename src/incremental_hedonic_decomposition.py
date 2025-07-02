"""Incremental hedonic decomposition of stranded farmland value.

Decomposes the hedonic stranded estimate by adding channels incrementally:
  Model 1: Climate only (tmax_july + tmax_july^2 + precip_growing + state FE)
  Model 2: + Demographics (log_pop + log_income)
  Model 3: + Soil proxy (max historical yield, normalized by crop)

Each model's stranded estimate is computed by applying CMIP6 SSP2-4.5 warming
deltas to the fitted model. Channel contributions = stranded_N - stranded_{N-1}.
The gap between Model 3 and DCF central ($105B) is the 'unmodeled channels' gap.

Output:
    results/decomposition/incremental_hedonic.json

Author: Keshav Krishnan | 2026-03-19
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

# Growing season months April–September
GROWING_MONTHS = [4, 5, 6, 7, 8, 9]

# CPI deflators (2023 USD)
CPI_2022 = 296.8
CPI_2023 = 304.7
DEFLATOR_2022 = CPI_2023 / CPI_2022

# Winsorize land values at 1st/99th pctile
LAND_VALUE_UPPER_PCTILE = 99
LAND_VALUE_LOWER_PCTILE = 1

# DCF central estimate (from state/headline_numbers.json)
DCF_CENTRAL_B = 105.1

# USDA Census of Agriculture 2022: total acres in farms by state FIPS
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


def build_soil_proxy(nass_yields: pd.DataFrame) -> pd.DataFrame:
    """Build soil quality proxy from max historical (pre-2010) yield by county.

    For each county, compute the crop-normalized maximum yield across all
    crops in the pre-2010 training period. This captures inherent soil
    productivity that's correlated with land value but not captured by
    climate or demographic controls.

    Strategy:
      1. Filter NASS to 1990-2009 (historical, training-split era).
      2. Max yield per county-crop (peak capacity, not average).
      3. Z-score each crop's max yield across counties (removes unit differences
         between corn bu/ac, cotton lb/ac, etc.).
      4. Take the mean z-score across crops for each county → single soil index.

    Args:
        nass_yields: NASS county yields (fips, year, crop, yield_bu_acre).

    Returns:
        DataFrame with fips and soil_index (z-scored, mean 0, std ~1).
    """
    logger.info("Building soil quality proxy from historical yields...")

    hist = nass_yields[nass_yields['year'].between(1990, 2009)].copy()
    hist = hist.dropna(subset=['yield_bu_acre'])

    # Max yield per county-crop
    max_yield = (
        hist.groupby(['fips', 'crop'])['yield_bu_acre']
        .max()
        .reset_index()
        .rename(columns={'yield_bu_acre': 'max_yield'})
    )

    # Z-score within each crop (across counties)
    def zscore(x):
        """Z-score a series, return NaN if std is 0."""
        mu = x.mean()
        sd = x.std()
        if sd == 0 or np.isnan(sd):
            return x * 0
        return (x - mu) / sd

    max_yield['yield_z'] = max_yield.groupby('crop')['max_yield'].transform(zscore)

    # County-level mean z-score across crops (soil index)
    soil = (
        max_yield.groupby('fips')['yield_z']
        .mean()
        .reset_index()
        .rename(columns={'yield_z': 'soil_index'})
    )

    logger.info(
        f"  Soil proxy: {len(soil)} counties, "
        f"mean={soil['soil_index'].mean():.3f}, "
        f"std={soil['soil_index'].std():.3f}"
    )
    return soil


def build_cross_section(
    land_values: pd.DataFrame,
    climate_monthly: pd.DataFrame,
    acs: pd.DataFrame,
    nass_yields: pd.DataFrame,
) -> pd.DataFrame:
    """Build county-level cross-section for incremental hedonic regression.

    Uses 2017/2022 land values (2023 USD), 2019-2023 climate averages,
    2019-2023 ACS demographics, and pre-2010 historical yield soil proxy.

    Args:
        land_values: NASS land values (fips, year, land_value_per_acre).
        climate_monthly: PRISM monthly climate (fips, year, tmax_m*, precip_m*).
        acs: ACS demographics (fips, year, total_population,
             median_household_income).
        nass_yields: NASS county yields (fips, year, crop, yield_bu_acre,
                     acres_harvested).

    Returns:
        DataFrame with one row per county: fips, state_fips, log_land_value,
        tmax_july, tmax_july_sq, precip_growing, log_pop, log_income,
        soil_index, farm_acres, land_value_per_acre.
    """
    logger.info("Building cross-section for incremental hedonic...")

    # --- Land value: 2017 and 2022, deflated to 2023 USD ---
    lv = land_values[land_values['year'].isin([2017, 2022])].copy()
    lv.loc[lv['year'] == 2022, 'land_value_per_acre'] *= DEFLATOR_2022
    lv_cs = (
        lv.groupby('fips')['land_value_per_acre']
        .mean()
        .reset_index()
    )
    lo = np.percentile(lv_cs['land_value_per_acre'], LAND_VALUE_LOWER_PCTILE)
    hi = np.percentile(lv_cs['land_value_per_acre'], LAND_VALUE_UPPER_PCTILE)
    lv_cs = lv_cs[lv_cs['land_value_per_acre'].between(lo, hi)].copy()
    logger.info(f"  Land values: {len(lv_cs)} counties (winsorized [{lo:.0f}, {hi:.0f}])")

    # --- Climate: 2019-2023 average ---
    clim = climate_monthly[climate_monthly['year'].between(2019, 2023)].copy()
    precip_cols = [f'precip_m{m:02d}' for m in GROWING_MONTHS]
    clim['precip_growing'] = clim[precip_cols].sum(axis=1)
    clim['tmax_july'] = clim['tmax_m07']
    clim_cs = (
        clim.groupby('fips')[['tmax_july', 'precip_growing']]
        .mean()
        .reset_index()
    )
    logger.info(f"  Climate: {len(clim_cs)} counties (2019-2023 avg)")

    # --- ACS: 2019-2023 average ---
    acs_w = acs[acs['year'].between(2019, 2023)].copy()
    acs_cs = (
