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
