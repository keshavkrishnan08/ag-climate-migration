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
