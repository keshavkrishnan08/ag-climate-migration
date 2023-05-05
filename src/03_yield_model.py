"""Phase 3A: Core yield trend model — LightGBM.

Predicts county-crop yield as a function of climate and technology features.
This is the engine of all projections.

Architecture (PRD Section 5.1):
    - LightGBM gradient boosted trees
    - Outcome: detrended yield anomaly (z-score)
    - After prediction: re-add projected technology trend
    - Final projected yield = tech_component + climate_component

Cross-Validation:
    STRICT TEMPORAL ROLLING CV — never shuffle
    Fold 1: train 1950-1985, val 1986-1990
    Fold 2: train 1950-1990, val 1991-1995
    Fold 3: train 1950-1995, val 1996-2000
    Fold 4: train 1950-2000, val 2001-2005
    Fold 5: train 1950-2005, val 2006-2010
    Final test: train 1950-2012, test 2013-2023

Performance thresholds (anomaly model — z-scored target):
    Spearman ≥ 0.35 overall (Schlenker & Roberts 2009 benchmark)
    Spearman ≥ 0.30 per crop (cotton excluded — driven by non-climate factors)
    R² ≥ 0.10 on z-scored anomalies (expected range ~0.15-0.25)
    Note: raw-yield R² thresholds (corn≥0.72 etc.) do NOT apply here —
    z-scored anomaly R² ~0.18 is literature-appropriate and not a model deficiency.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from scipy import stats
from loguru import logger
import yaml
import json
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from utils.validation import temporal_rolling_cv, check_no_future_leakage, compute_performance_metrics
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

RANDOM_SEED = CONFIG['yield_model']['random_seed']


def get_lgb_params() -> dict:
    """Get LightGBM parameters from config.

    Returns:
        Dict of LightGBM hyperparameters.
    """
    return {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': CONFIG['yield_model']['n_estimators'],
        'learning_rate': CONFIG['yield_model']['learning_rate'],
        'max_depth': CONFIG['yield_model']['max_depth'],
        'num_leaves': CONFIG['yield_model']['num_leaves'],
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_SEED,
        'verbose': -1,
    }


def get_feature_columns(df: pd.DataFrame) -> list:
    """Extract feature column names from the panel.

    Args:
        df: Feature matrix DataFrame.

    Returns:
        List of feature column names (excludes target and identifiers).
    """
    exclude = {
        'fips', 'year', 'crop', 'yield_bu_acre', 'yield_anomaly',
        'acres_harvested', 'production'
    }
    return [c for c in df.columns if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


def prepare_data(
    panel: pd.DataFrame,
    target_col: str = 'yield_anomaly'
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare feature matrix and target for modeling.

    Args:
        panel: Complete feature matrix.
        target_col: Name of target column.

    Returns:
        Tuple of (X features, y target, years array).
    """
    feature_cols = get_feature_columns(panel)
    logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")
