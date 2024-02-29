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
