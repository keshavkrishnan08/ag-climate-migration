"""Compute stranded agricultural assets under SSP3-7.0.

Runs the same conservative (ML only) and central (ML + SR + indirect) methods
as 06_stranded.py but using SSP370 yield and climate projections.
Reports results and compares against the SSP245 baseline.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
PROJECTIONS_DIR = PROJECT_ROOT / 'data' / 'projections'
RESULTS_DIR = PROJECT_ROOT / 'results'
OUTPUT_DIR = RESULTS_DIR / 'stranded_assets'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

SCENARIO = 'SSP370'

# ── Inline helper functions (mirrors 06_stranded.py) ─────────────────────────

COMMODITY_PRICES = {
    'corn': 5.50, 'soybeans': 12.80, 'wheat_winter': 7.20,
    'wheat_spring': 8.10, 'cotton': 0.78, 'sorghum': 5.30,
    'barley': 6.10, 'oats': 3.80,
}

SR_THRESHOLD_MODERATE = 29.0
SR_JULY_DAYS = 31
SR_SHOULDER_DAYS = 60
SR_COEFFICIENTS = {
    'corn':         -0.0662,
    'soybeans':     -0.0560,
    'wheat_winter': -0.0420,
    'wheat_spring': -0.0420,
    'cotton':       -0.0662,
    'sorghum':      -0.0662,
    'barley':       -0.0420,
    'oats':         -0.0420,
}


def compute_stranded_vectorized(
    yield_proj: pd.DataFrame,
    land_values: pd.DataFrame,
    discount_rate: float = 0.04,
    horizon: int = 30,
    scenario: str = 'SSP370'
) -> pd.DataFrame:
    """Compute stranded assets (ML model only) across all counties/crops.

    Args:
        yield_proj: Projections DataFrame with yield_tech_trend, climate_impact_bu, acres_harvested.
        land_values: NASS land values with fips, land_value_per_acre.
        discount_rate: Real discount rate.
        horizon: Projection horizon in years.
        scenario: Climate scenario label.

    Returns:
        DataFrame with stranded value per county (aggregated across crops).
