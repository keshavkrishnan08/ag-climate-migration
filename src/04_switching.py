"""Phase 3B: Crop switching model — the adaptation model.

Predicts the probability of a county switching from crop A to crop B
given climate and economic conditions.

Architecture (PRD Section 5.2):
    - Binary classifier for each switching pair
    - Outcome: P(switch from corn to sorghum in county c, year t)
    - Features: multi-year temperature trend, relative profitability,
      neighbor counties that already switched, farm debt
    - Implementation: LightGBM classifier
    - Calibration: Platt scaling for valid probabilities
    - CRITICAL: switching probability must be monotone in temp trend
      (hotter → more likely to switch away from heat-sensitive crops)

Switching pairs (from config):
    - [corn, soybeans]
    - [corn, sorghum]
    - [cotton, soybeans]
    - [wheat_winter, wheat_spring]
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    brier_score_loss, log_loss
)
from scipy import stats
from loguru import logger
import yaml
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

RANDOM_SEED = CONFIG['yield_model']['random_seed']
SWITCHING_PAIRS = [tuple(p) for p in CONFIG['crops']['switching_pairs']]


def build_switching_labels(
    panel: pd.DataFrame,
    switching_rates: pd.DataFrame,
    pair: Tuple[str, str],
    threshold: float = 0.05
) -> pd.DataFrame:
    """Build binary switching labels for a crop pair.

    Args:
        panel: Feature matrix with county-year observations.
        switching_rates: County-year switching rate data.
        pair: Tuple of (from_crop, to_crop).
        threshold: Fraction above which we label as 'switched'.

    Returns:
        DataFrame with features and binary switch label.
    """
    from_crop, to_crop = pair

    # Get counties that grew from_crop
    from_counties = panel[panel['crop'] == from_crop][['fips', 'year']].copy()

    # Merge with switching rates
    if not switching_rates.empty:
        from_counties = from_counties.merge(
            switching_rates[['fips', 'year', f'switch_{from_crop}_to_{to_crop}']],
            on=['fips', 'year'],
            how='left'
