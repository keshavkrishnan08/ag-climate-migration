"""Fix 5 — Validate crop switching model against 4 historical events.

Uses NASS county acreage data (1950+) and nClimDiv climate data to test
whether a simple climate-feature LightGBM model can reproduce observed
crop switching patterns. Four validation events from SWARM spec §A3:

1. Sorghum expansion in southern Plains (1950-1975) — POSITIVE
2. Cotton retreat from Missouri/Tennessee (1980-2010) — POSITIVE
3. Winter wheat boundary shift in Kansas (1990-2010) — POSITIVE
4. Soybean adoption in Corn Belt (1960-1980) — NEGATIVE test

All four must pass before the switching model can be trusted for projections.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NASS_PATH = PROJECT_ROOT / "data" / "raw" / "nass" / "nass_county_yields.parquet"
CLIMATE_PATH = PROJECT_ROOT / "data" / "raw" / "prism" / "county_climate_annual.parquet"
CLIMATE_MONTHLY_PATH = PROJECT_ROOT / "data" / "raw" / "prism" / "county_climate_monthly.parquet"
OUTPUT_PATH = PROJECT_ROOT / "state" / "validation" / "historical_switching.json"

RANDOM_SEED = 42

# Kansas county centroid latitudes (FIPS -> latitude in degrees N).
# Source: US Census Bureau TIGER/Gazetteer county centroids.
KANSAS_COUNTY_LAT = {
    "20001": 37.88, "20003": 37.77, "20005": 38.91, "20007": 37.25,
