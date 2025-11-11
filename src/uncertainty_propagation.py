"""Monte Carlo uncertainty propagation for stranded asset DCF estimate.

Propagates yield model uncertainty (R²=0.21) through the DCF stranded asset
computation using 1,000 Monte Carlo draws from the residual distribution.

Outputs:
    results/stranded_assets/uncertainty_propagation.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
RESULTS_DIR = PROJECT_ROOT / 'results'
PROJECTIONS_DIR = PROJECT_ROOT / 'data' / 'projections'

# Yield model test R² from v2 model (results/yield_model_v2_metrics.json)
R2_YIELD = 0.2059691810308517

# Commodity prices (2023 USD) — must match 06_stranded.py
COMMODITY_PRICES = {
    'corn': 5.50, 'soybeans': 12.80, 'wheat_winter': 7.20,
    'wheat_spring': 8.10, 'cotton': 0.78, 'sorghum': 5.30,
    'barley': 6.10, 'oats': 3.80,
}

# DCF parameters: baseline (4%, 30yr) — conservative scenario
DISCOUNT_RATE = 0.04
HORIZON = 30
N_ITER = 1000
SEED = 42


def compute_total_stranded(yield_proj: pd.DataFrame) -> float:
    """Compute total national stranded value ($ billions) from yield projections.

    Args:
        yield_proj: DataFrame with climate_impact_bu, acres_harvested, year, crop.

    Returns:
        Total stranded value in billions USD (positive = loss).
    """
    df = yield_proj.copy()
    df['price'] = df['crop'].map(COMMODITY_PRICES).fillna(5.0)
