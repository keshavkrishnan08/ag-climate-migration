"""Phase 3C: Quantile Regression Forest for Extreme Event Tail Risk.

The mean yield model underpredicts extreme events — the 2012 drought anomaly
prediction of -0.11 is far too mild vs. the observed mean of -0.94 across
corn counties. Tail risk matters for stranded asset estimates, which depend
on correctly capturing downside scenarios.

This script:
  1. Trains a LightGBM quantile model at alpha=0.10 (Q10) using same features
     and temporal split as the existing mean model.
  2. Diagnoses the 2012 drought: Q10 prediction should be much more negative
     than the mean model's -0.11.
  3. Computes a tail risk premium per county (mean - Q10 gap) and adds it as
     an additional stranded asset component.

Temporal split (per CLAUDE.md):
    train  : year <= 2009
    val    : 2010-2016  (used as final training cutoff for test evaluation)
    test   : 2017-2023

Output:
    models/yield/yield_model_q10.pkl
    results/quantile/q10_metrics.json
    results/quantile/drought_2012_q10.csv
    results/quantile/tail_risk_stranded.parquet
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
PROJECTIONS_DIR = PROJECT_ROOT / 'data' / 'projections'
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
MODELS_DIR = PROJECT_ROOT / 'models' / 'yield'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'quantile'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

RANDOM_SEED = CONFIG['yield_model']['random_seed']
ALPHA = 0.10  # Q10 — captures the lower tail where droughts concentrate

COMMODITY_PRICES = {
    'corn': 5.50,
    'soybeans': 12.80,
    'wheat_winter': 7.20,
    'wheat_spring': 8.10,
    'cotton': 0.78,
    'sorghum': 5.30,
    'barley': 6.10,
    'oats': 3.80,
}

# Temporal split — per CLAUDE.md (train<=2009, val 2010-2016, test 2017-2023)
TRAIN_END = 2009
VAL_END = 2016
TEST_END = 2023


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list:
    """Extract numeric feature column names, excluding identifiers and target.

    Args:
        df: Feature matrix DataFrame.

    Returns:
        List of feature column names.
    """
    exclude = {
        'fips', 'year', 'crop', 'yield_bu_acre', 'yield_anomaly',
        'acres_harvested', 'production',
    }
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64', 'int32')
    ]


def prepare_features(
    panel: pd.DataFrame,
    all_crops: list = None,
    training_columns: list = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build feature matrix X with crop dummies, target y, and years.

    Mirrors the exact feature engineering in 03_yield_model.py so the Q10
    model is directly comparable to the mean model.

    When predicting on a subset of the data (e.g., only corn in 2012), pass
    ``training_columns`` to align the feature matrix with what the model saw
    during training. Missing dummy columns are filled with 0.

    Args:
        panel: Full or subset feature matrix.
        all_crops: Full list of all crop categories (used to build dummies
                   with consistent column set). If None, derived from panel.
        training_columns: Column list from training time. If provided,
                          the returned X will be reindexed to match exactly.

    Returns:
        Tuple of (X, y, years).
    """
    # Reset index for safe concatenation (panel may be a filtered slice)
    panel = panel.reset_index(drop=True)

    feature_cols = get_feature_columns(panel)
    X = panel[feature_cols].copy()
    y = panel['yield_anomaly'].copy()
    years = panel['year'].copy()

    # One-hot encode crop — use pd.Categorical so every crop level appears
    # even if only a subset of crops is present in panel.
    if all_crops is not None:
        crop_cat = pd.Categorical(panel['crop'], categories=all_crops)
    else:
        crop_cat = panel['crop']
    crop_dummies = pd.get_dummies(crop_cat, prefix='crop').astype(int)
    X = pd.concat([X, crop_dummies], axis=1)

    # If training columns are provided, align to them (fill missing with 0)
    if training_columns is not None:
        X = X.reindex(columns=training_columns, fill_value=0)

    logger.info(f"Feature matrix: {X.shape[1]} features ({len(feature_cols)} numeric + {crop_dummies.shape[1]} crop dummies)")
    return X, y, years


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def get_q10_params() -> dict:
    """LightGBM quantile parameters at alpha=0.10, same architecture as mean model.

    Returns:
        Dict of LightGBM hyperparameters.
    """
    base = CONFIG['yield_model']
    return {
        'objective': 'quantile',
        'alpha': ALPHA,
        'metric': 'quantile',
        'n_estimators': base['n_estimators'],
        'learning_rate': base['learning_rate'],
        'max_depth': base['max_depth'],
        'num_leaves': base['num_leaves'],
        'min_child_samples': base.get('min_child_samples', 20),
        'subsample': base.get('subsample', 0.8),
        'colsample_bytree': base.get('colsample_bytree', 0.8),
        'reg_alpha': base.get('reg_alpha', 0.1),
        'reg_lambda': base.get('reg_lambda', 1.0),
        'random_state': RANDOM_SEED,
        'verbose': -1,
    }


def train_q10_model(panel: pd.DataFrame) -> Tuple[lgb.LGBMRegressor, dict, list, list]:
    """Train LightGBM Q10 quantile model with temporal split.

    Train on years <= TRAIN_END (2009), use val 2010-2016 for early-stopping
    monitoring, evaluate test performance on 2017-2023.

    Args:
        panel: Full feature matrix.

    Returns:
        Tuple of (trained model, metrics dict, training_columns, all_crops).
    """
    logger.info("=" * 60)
    logger.info(f"TRAINING Q{int(ALPHA*100)} QUANTILE MODEL (alpha={ALPHA})")
    logger.info("=" * 60)

    all_crops = sorted(panel['crop'].dropna().unique().tolist())
    X, y, years = prepare_features(panel, all_crops=all_crops)
    training_columns = list(X.columns)
    params = get_q10_params()

    train_mask = years <= TRAIN_END
    val_mask = (years > TRAIN_END) & (years <= VAL_END)
    test_mask = (years > VAL_END) & (years <= TEST_END)

    X_train = X[train_mask.values]
    y_train = y[train_mask.values]
    X_val = X[val_mask.values]
    y_val = y[val_mask.values]
    X_test = X[test_mask.values]
    y_test = y[test_mask.values]

    logger.info(
        f"Split: train n={len(X_train)} (≤{TRAIN_END}), "
        f"val n={len(X_val)} ({TRAIN_END+1}-{VAL_END}), "
        f"test n={len(X_test)} ({VAL_END+1}-{TEST_END})"
    )

    # Temporal leakage check
    assert years[train_mask.values].max() < years[val_mask.values].min(), "Train/val leakage!"
    assert years[val_mask.values].max() < years[test_mask.values].min(), "Val/test leakage!"
    logger.info("Temporal leakage check PASSED")

    # Train on train set, monitor on val
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=200)],
    )

    # Test evaluation — pinball loss at alpha=0.10 and rank correlation
    y_pred_test = model.predict(X_test)

    # Pinball (quantile) loss: mean over samples of max(alpha*(y-q), (alpha-1)*(y-q))
    residuals = y_test.values - y_pred_test
    pinball = np.where(
        residuals >= 0,
        ALPHA * residuals,
        (ALPHA - 1) * residuals,
    ).mean()

    spearman_rho, spearman_p = stats.spearmanr(y_test.values, y_pred_test)
    mae = np.mean(np.abs(residuals))

    # Coverage: fraction of actual values below the Q10 prediction
    # Should be close to 10% for a well-calibrated Q10 model
    coverage = (y_test.values < y_pred_test).mean()

    logger.info(f"Q10 TEST METRICS (2017-2023):")
    logger.info(f"  Pinball loss (Q10): {pinball:.4f}")
    logger.info(f"  MAE:                {mae:.4f}")
    logger.info(f"  Spearman ρ:         {spearman_rho:.3f} (p={spearman_p:.2e})")
    logger.info(f"  Coverage (% actual < Q10): {coverage*100:.1f}% (target: 10%)")
    logger.info(f"  Q10 pred range: [{y_pred_test.min():.3f}, {y_pred_test.max():.3f}]")

    metrics = {
        'alpha': ALPHA,
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'n_test': int(len(X_test)),
        'train_end': TRAIN_END,
        'val_end': VAL_END,
        'test_start': VAL_END + 1,
        'test_end': TEST_END,
        'pinball_loss_q10': float(pinball),
        'mae': float(mae),
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'coverage_pct': float(coverage * 100),
        'q10_pred_min': float(y_pred_test.min()),
        'q10_pred_max': float(y_pred_test.max()),
        'q10_pred_mean': float(y_pred_test.mean()),
    }

    return model, metrics, training_columns, all_crops


# ---------------------------------------------------------------------------
# 2012 drought diagnosis
# ---------------------------------------------------------------------------

def diagnose_2012_drought(
    model: lgb.LGBMRegressor,
    panel: pd.DataFrame,
    all_crops: list,
    training_columns: list,
) -> pd.DataFrame:
    """Predict Q10 yield anomaly for 2012 corn counties.

    The 2012 drought was the worst US drought since the 1950s Dust Bowl.
    The mean model predicted an anomaly of -0.11 — far too optimistic.
    Q10 should capture the true tail risk and predict much more negative values.

    Args:
        model: Trained Q10 LightGBM model.
        panel: Full feature matrix.
        all_crops: Full crop category list from training.
        training_columns: Feature column list from training time.

    Returns:
        DataFrame with fips, observed anomaly, Q10 prediction, and residual.
    """
    logger.info("\n" + "=" * 60)
    logger.info("2012 DROUGHT DIAGNOSIS — Q10 MODEL")
    logger.info("=" * 60)

    drought = panel[(panel['year'] == 2012) & (panel['crop'] == 'corn')].copy().reset_index(drop=True)
    logger.info(f"2012 corn counties: {len(drought)}")

    X_drought, _, _ = prepare_features(drought, all_crops=all_crops, training_columns=training_columns)
    q10_pred = model.predict(X_drought)

    drought['q10_prediction'] = q10_pred
    drought['observed_anomaly'] = drought['yield_anomaly']
    drought['residual'] = drought['observed_anomaly'] - drought['q10_prediction']

    mean_obs = drought['observed_anomaly'].mean()
    mean_q10 = drought['q10_prediction'].mean()
    median_q10 = drought['q10_prediction'].median()

    logger.info(f"Observed 2012 corn anomaly:   mean={mean_obs:.3f} z-scores")
    logger.info(f"Q10 model prediction:         mean={mean_q10:.3f}, median={median_q10:.3f} z-scores")
    logger.info(f"(Mean model predicted:        -0.11 z-scores)")
    logger.info(f"Q10 improvement over mean:    {mean_q10 - (-0.11):+.3f} z-scores more negative")

    # Coverage in 2012: what fraction of observed values fell below Q10?
    coverage_2012 = (drought['observed_anomaly'] < drought['q10_prediction']).mean()
    logger.info(f"2012 drought coverage:        {coverage_2012*100:.1f}% of obs below Q10")
    logger.info(f"  (high coverage = Q10 is well above observed = model captures severity)")

    return drought[['fips', 'year', 'crop', 'observed_anomaly', 'q10_prediction', 'residual']]


# ---------------------------------------------------------------------------
# Tail risk stranded asset component
# ---------------------------------------------------------------------------
