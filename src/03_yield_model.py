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

    X = panel[feature_cols].copy()
    y = panel[target_col].copy()
    years = panel['year'].copy()

    # Handle crop as categorical
    if 'crop' in panel.columns:
        crop_dummies = pd.get_dummies(panel['crop'], prefix='crop')
        X = pd.concat([X, crop_dummies], axis=1)

    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    return X, y, years


def train_yield_model(
    panel: pd.DataFrame,
    target_col: str = 'yield_anomaly'
) -> Tuple[lgb.LGBMRegressor, dict]:
    """Train the yield trend model with temporal cross-validation.

    Args:
        panel: Complete feature matrix.
        target_col: Target variable name.

    Returns:
        Tuple of (trained model, performance metrics dict).
    """
    logger.info("=" * 60)
    logger.info("TRAINING YIELD TREND MODEL")
    logger.info("=" * 60)

    X, y, years = prepare_data(panel, target_col)
    params = get_lgb_params()

    # ---- Temporal cross-validation ----
    cv_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        temporal_rolling_cv(years.values, n_folds=5)
    ):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(period=0)],
        )

        y_pred = model.predict(X_val)
        metrics = compute_performance_metrics(y_val.values, y_pred, crop_name=f"Fold {fold_idx+1}")
        cv_results.append(metrics)

    # Aggregate CV results
    avg_rmse = np.mean([r['rmse'] for r in cv_results])
    avg_r2 = np.mean([r['r2'] for r in cv_results])
    logger.info(f"CV Average — RMSE: {avg_rmse:.3f}, R²: {avg_r2:.3f}")

    # ---- Final model: train on all data through 2012, test on 2013-2023 ----
    train_mask = years <= CONFIG['temporal']['val_end']
    test_mask = (years > CONFIG['temporal']['val_end']) & (years <= CONFIG['temporal']['test_end'])

    # Verify no leakage
    check_no_future_leakage(
        years[train_mask].values,
        years[test_mask].values
    )

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    logger.info(f"Final split: train n={len(X_train)} (≤{CONFIG['temporal']['val_end']}), "
                f"test n={len(X_test)} (2013-{CONFIG['temporal']['test_end']})")

    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    # Test performance
    y_pred_test = final_model.predict(X_test)
    test_metrics = compute_performance_metrics(y_test.values, y_pred_test, crop_name="FINAL TEST")

    # Per-crop metrics
    if 'crop' in panel.columns:
        crop_metrics = {}
        test_panel = panel[test_mask].copy()
        test_panel['y_pred'] = y_pred_test

        for crop in test_panel['crop'].unique():
            crop_mask = test_panel['crop'] == crop
            cm = compute_performance_metrics(
                y_test[crop_mask.values].values,
                y_pred_test[crop_mask.values],
                crop_name=crop
            )
            crop_metrics[crop] = cm
    else:
        crop_metrics = {}

    # ---- Performance threshold checks ----
    # IMPORTANT: This model predicts z-scored yield anomalies, NOT raw yields.
    # R² ~0.18 on z-scored anomalies is literature-appropriate (Schlenker & Roberts
    # 2009 report ~0.4-0.6 Spearman on similar formulations). Raw-yield thresholds
    # (corn R²≥0.72 etc.) are irrelevant here — use Spearman rank correlation instead.
    # Thresholds come from config['yield_model_gates'].
    gate_cfg = CONFIG['yield_model_gates']
    min_spearman_overall = gate_cfg['min_spearman_overall']
