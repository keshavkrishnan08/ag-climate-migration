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
