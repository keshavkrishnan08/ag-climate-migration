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
        )
        switch_col = f'switch_{from_crop}_to_{to_crop}'
        from_counties['switched'] = (from_counties[switch_col] > threshold).astype(int)
    else:
        from_counties['switched'] = 0

    return from_counties


def get_switching_features(
    panel: pd.DataFrame,
    pair: Tuple[str, str]
) -> List[str]:
    """Get relevant features for a switching model.

    Args:
        panel: Feature matrix.
        pair: Crop pair being modeled.

    Returns:
        List of feature column names.
    """
    exclude = {
        'fips', 'year', 'crop', 'yield_bu_acre', 'yield_anomaly',
        'acres_harvested', 'production', 'switched'
    }
    base_features = [c for c in panel.columns
                     if c not in exclude and panel[c].dtype in ('float64', 'float32', 'int64')]

    # Add pair-specific features
    # - Relative profitability between crops
    # - Neighbor county switching rate
    # - Temperature trend (critical for monotonicity)
    return base_features


def train_switching_model(
    panel: pd.DataFrame,
    switching_rates: pd.DataFrame,
    pair: Tuple[str, str]
) -> Tuple[CalibratedClassifierCV, dict]:
    """Train a calibrated switching probability model for one crop pair.

    Args:
        panel: Feature matrix.
        switching_rates: County-year switching rates.
        pair: (from_crop, to_crop) tuple.

    Returns:
        Tuple of (calibrated model, metrics dict).
    """
    from_crop, to_crop = pair
    logger.info(f"Training switching model: {from_crop} → {to_crop}")

    # Build labels
    data = build_switching_labels(panel, switching_rates, pair)

    if data.empty or data['switched'].sum() == 0:
        logger.warning(f"No switching events for {from_crop} → {to_crop}")
        return None, {}
