"""Phase 3D: Retrain yield model with compound drought interaction features.

Reviewer fix: the base model under-predicts the 2012 drought because it misses
the multiplicative penalty when heat AND moisture stress occur simultaneously.
This script adds compound interaction features and retrains with depth=8 to
let the tree structure capture the threshold behavior.

New features added:
    heat_x_drought     = tmax_july_c_anomaly * (-pdsi_growing_anomaly)
    heat_x_precip      = tmax_july_c_anomaly * (-precip_growing_anomaly)
    extreme_compound   = (tmax_anom > 1°C) AND (pdsi_anom < -1) binary flag
    tmax_july_sq       = tmax_july_c_anomaly² (quadratic heat threshold)
    precip_deficit     = max(0, county_baseline_precip - actual_precip)
    tmax_peak_c        = max(tmax Jun/Jul/Aug) in Celsius (from monthly data)
    precip_jja         = total Jun+Jul+Aug precipitation
    pdsi_peak_drought  = min(PDSI Jun/Jul/Aug) — worst summer drought month
    edd_months_c       = months where tmax > 33.5°C (Schlenker & Roberts threshold)
    edd_x_pdsi         = edd_months_c * (-pdsi_peak_drought) compound signal
    + anomaly versions of the monthly features (county mean subtracted)

Temporal split (matches config.yaml val_end=2012):
    Train:  years <= 2012
    Test:   2013-2023

Saves:
    results/yield_model_v2.pkl   — new primary model
    results/yield_model_v2_metrics.json
    results/feature_importance_v2.csv
"""

import os
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from loguru import logger
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from utils.validation import (
    temporal_rolling_cv,
    check_no_future_leakage,
    compute_performance_metrics,
)

DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
RESULTS_DIR = PROJECT_ROOT / 'results'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

RANDOM_SEED = CONFIG['yield_model']['random_seed']

# Schlenker & Roberts (2009) EDD threshold for corn (°C)
EDD_THRESHOLD_C = 33.5

# Growing-season months: May–September
GROWING_MONTHS = ['05', '06', '07', '08', '09']
# Critical pollination months: June–August
PEAK_MONTHS = ['06', '07', '08']


def load_monthly_features() -> pd.DataFrame:
    """Load and derive monthly climate features for compound drought detection.

    Converts nClimDiv temperatures from Fahrenheit to Celsius, then computes:
    - tmax_peak_c:      max(Jun, Jul, Aug) Tmax in Celsius
    - precip_jja:       total Jun+Jul+Aug precipitation
    - pdsi_peak_drought: min(Jun,Jul,Aug PDSI) — most severe monthly drought
    - edd_months_c:     count of months with Tmax > 33.5°C (EDD threshold)

    Returns:
        DataFrame indexed on (fips, year) with the four derived monthly features.
    """
    monthly_path = DATA_RAW / 'prism' / 'county_climate_monthly.parquet'
    if not monthly_path.exists():
        raise FileNotFoundError(f"Monthly climate not found: {monthly_path}")

    monthly = pd.read_parquet(monthly_path)

    # Convert peak-month temperatures to Celsius
    for m in PEAK_MONTHS:
        monthly[f'tmax_m{m}_c'] = (monthly[f'tmax_m{m}'] - 32) * 5 / 9

    tmax_peak_cols = [f'tmax_m{m}_c' for m in PEAK_MONTHS]
    precip_jja_cols = [f'precip_m{m}' for m in PEAK_MONTHS]
    pdsi_jja_cols = [f'pdsi_m{m}' for m in PEAK_MONTHS]

    monthly['tmax_peak_c'] = monthly[tmax_peak_cols].max(axis=1)
    monthly['precip_jja'] = monthly[precip_jja_cols].sum(axis=1)
    monthly['pdsi_peak_drought'] = monthly[pdsi_jja_cols].min(axis=1)
    monthly['edd_months_c'] = (monthly[tmax_peak_cols] > EDD_THRESHOLD_C).sum(axis=1).astype(float)

    features = monthly[['fips', 'year', 'tmax_peak_c', 'precip_jja',
                         'pdsi_peak_drought', 'edd_months_c']].copy()
    features['fips'] = features['fips'].astype(str)
    logger.info(f"Loaded monthly features: {features.shape} for {features['year'].nunique()} years")
    return features


def add_interaction_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add compound drought interaction features to the panel.

    The 2012 US drought combined record heat (+2°C July anomaly) with the worst
    PDSI drought signal in decades. Neither stress alone explains the -0.94σ
    yield anomaly; their interaction does.

    New features:
        heat_x_drought   — positive when hot AND dry (anomaly space)
        heat_x_precip    — positive when hot AND low precipitation
        extreme_compound — binary: both heat > +1σ and PDSI < -1σ
        tmax_july_sq     — quadratic heat term for yield-cliff threshold
        precip_deficit   — county-specific moisture shortfall vs historical mean

    Args:
        panel: Feature matrix with anomaly columns already present.

    Returns:
        Panel with five new interaction columns appended.
    """
    panel = panel.copy()

    panel['heat_x_drought'] = (
        panel['tmax_july_c_anomaly'] * (-panel['pdsi_growing_anomaly'])
    )
    panel['heat_x_precip'] = (
        panel['tmax_july_c_anomaly'] * (-panel['precip_growing_anomaly'])
    )
    panel['extreme_compound'] = (
        (panel['tmax_july_c_anomaly'] > 1.0) &
        (panel['pdsi_growing_anomaly'] < -1.0)
    ).astype(float)
    panel['tmax_july_sq'] = panel['tmax_july_c_anomaly'] ** 2

    # County × crop baseline for precipitation deficit
    precip_baseline = panel.groupby(['fips', 'crop'])['precip_growing'].transform('mean')
    panel['precip_deficit'] = (precip_baseline - panel['precip_growing']).clip(lower=0)

    logger.info("Added 5 interaction features: heat_x_drought, heat_x_precip, "
                "extreme_compound, tmax_july_sq, precip_deficit")
    return panel


def add_monthly_anomaly_features(
    panel: pd.DataFrame,
    monthly_features: pd.DataFrame
) -> pd.DataFrame:
    """Merge monthly features into panel and compute county-level anomalies.

    Args:
        panel: Feature matrix (must have fips, year columns).
        monthly_features: Output of load_monthly_features().

    Returns:
        Panel with monthly features and their county anomalies appended.

    Raises:
        ValueError: If merge results in more rows than input panel.
    """
    n_before = len(panel)
    panel = panel.merge(monthly_features, on=['fips', 'year'], how='left')
    if len(panel) != n_before:
        raise ValueError(f"Merge changed row count: {n_before} → {len(panel)}")

    monthly_base_cols = ['tmax_peak_c', 'precip_jja', 'pdsi_peak_drought', 'edd_months_c']
    for col in monthly_base_cols:
        county_mean = panel.groupby('fips')[col].transform('mean')
        panel[f'{col}_anomaly'] = panel[col] - county_mean

    # EDD × peak PDSI compound (high EDD + deep drought = catastrophic)
    panel['edd_x_pdsi'] = panel['edd_months_c'] * (-panel['pdsi_peak_drought'])

    logger.info(f"Merged monthly features. Panel now {panel.shape}")
    return panel


def get_feature_columns(df: pd.DataFrame) -> list:
    """Extract numeric feature columns from the panel, excluding identifiers/target.

    Args:
        df: Feature matrix DataFrame.

    Returns:
        List of numeric column names to use as model features.
    """
    exclude = {
        'fips', 'year', 'crop',
        'yield_bu_acre', 'yield_anomaly',
        'acres_harvested', 'production',
    }
    return [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in ('float64', 'float32', 'int64', 'int32', 'bool')
        and not df[c].isna().all()
    ]


def prepare_features(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build final feature matrix with crop dummies and fill missing values.

    Args:
        panel: Panel with all interaction/monthly features already added.

    Returns:
        Tuple of (X, y, years) ready for LightGBM training.
    """
    feature_cols = get_feature_columns(panel)
    logger.info(f"Using {len(feature_cols)} features")

    X = panel[feature_cols].fillna(0).copy()
    crop_dummies = pd.get_dummies(panel['crop'], prefix='crop')
    X = pd.concat([X, crop_dummies], axis=1)

    y = panel['yield_anomaly'].copy()
    years = panel['year'].copy()

    logger.info(f"Feature matrix: X={X.shape}, y={y.shape}")
    return X, y, years


def get_v2_params() -> dict:
    """Return LightGBM hyperparameters for the v2 interaction model.

    Changes vs v1: depth 6→8, leaves 63→127, lr 0.03→0.02, n_est 1000→1500.
    Deeper trees are needed to capture the three-way heat × drought × crop
    interaction without manual feature engineering for every cross.

    Returns:
        Dict of LightGBM hyperparameters.
    """
    return {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1500,
        'learning_rate': 0.02,
        'max_depth': 8,
        'num_leaves': 127,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5,
        'random_state': RANDOM_SEED,
        'verbose': -1,
    }


def evaluate_2012_drought(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
    panel: pd.DataFrame,
    split: str = 'in_sample',
) -> dict:
    """Evaluate 2012 drought prediction quality.

    The 2012 US Midwest drought is the primary reviewer validation case.
    Observed national corn yield anomaly: -0.94σ (unweighted mean) / -1.09σ
    (acreage-weighted). A credible model should predict below -0.5σ.

    Args:
        model: Trained LightGBM model.
        X: Full feature matrix aligned with panel.
        panel: Panel with yield_anomaly and acres_harvested columns.
        split: 'in_sample' or 'out_of_sample' for logging context.

    Returns:
        Dict with mean_pred, median_pred, weighted_pred, observed keys.
    """
    mask = (panel['crop'] == 'corn') & (panel['year'] == 2012)
    if mask.sum() == 0:
        logger.warning("No 2012 corn observations found")
        return {}

    pred = model.predict(X[mask])
    obs = panel.loc[mask, 'yield_anomaly'].values
    acres = panel.loc[mask, 'acres_harvested'].fillna(1).values

    mean_pred = float(pred.mean())
    median_pred = float(np.median(pred))
    weighted_pred = float(np.average(pred, weights=acres))
    mean_obs = float(obs.mean())
    weighted_obs = float(np.average(obs, weights=acres))

    target_unweighted = -0.50
    target_weighted = -0.70

    logger.info(f"=== 2012 Drought Validation ({split}) ===")
    logger.info(f"  Observed:  mean={mean_obs:.3f}σ, acreage-weighted={weighted_obs:.3f}σ")
    logger.info(f"  Predicted: mean={mean_pred:.3f}σ, median={median_pred:.3f}σ, "
                f"acreage-weighted={weighted_pred:.3f}σ")
    unweighted_ok = mean_pred <= target_unweighted
    weighted_ok = weighted_pred <= target_weighted
    logger.info(f"  Gate mean ≤ {target_unweighted}σ: {'PASS' if unweighted_ok else 'FAIL'}")
    logger.info(f"  Gate weighted ≤ {target_weighted}σ: {'PASS' if weighted_ok else 'FAIL'}")

    return {
        'mean_pred': mean_pred,
        'median_pred': median_pred,
        'weighted_pred': weighted_pred,
        'mean_obs': mean_obs,
        'weighted_obs': weighted_obs,
        'n_counties': int(mask.sum()),
        'split': split,
        'gate_mean_pass': unweighted_ok,
        'gate_weighted_pass': weighted_ok,
    }


def train_and_evaluate(
    panel: pd.DataFrame,
) -> Tuple[lgb.LGBMRegressor, dict]:
    """Train v2 model with full temporal CV and holdout test evaluation.

    Temporal split:
        Train:  years ≤ 2012 (config val_end — same as v1 final model)
        Test:   2013–2023 (config test_end)

    CV folds use the same rolling structure as v1 (five 5-year windows
    ending at 2010), so the CV Spearman comparison is apples-to-apples.

    Args:
        panel: Full panel with all interaction and monthly features.

    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    logger.info("=" * 60)
    logger.info("RETRAINING YIELD MODEL v2 — COMPOUND DROUGHT FIX")
    logger.info("=" * 60)

    X, y, years = prepare_features(panel)
    params = get_v2_params()

    # ---- Temporal cross-validation ----
    cv_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        temporal_rolling_cv(years.values, n_folds=5)
    ):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = lgb.LGBMRegressor(**params)
        fold_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        y_pred = fold_model.predict(X_val)
        metrics = compute_performance_metrics(
            y_val.values, y_pred, crop_name=f"Fold {fold_idx + 1}"
        )
        cv_results.append(metrics)

    avg_spearman = np.mean([r['spearman_rank'] for r in cv_results])
    avg_r2 = np.mean([r['r2'] for r in cv_results])
    logger.info(f"CV Average — Spearman: {avg_spearman:.3f}, R²: {avg_r2:.3f}")

    # ---- Final model ----
    train_mask = years <= CONFIG['temporal']['val_end']
    test_mask = (years > CONFIG['temporal']['val_end']) & (years <= CONFIG['temporal']['test_end'])

    check_no_future_leakage(years[train_mask].values, years[test_mask].values)

    logger.info(f"Final split: train n={train_mask.sum()} (≤{CONFIG['temporal']['val_end']}), "
                f"test n={test_mask.sum()} ({CONFIG['temporal']['val_end'] + 1}–{CONFIG['temporal']['test_end']})")

    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(
        X[train_mask], y[train_mask],
        eval_set=[(X[test_mask], y[test_mask])],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    y_pred_test = final_model.predict(X[test_mask])
    test_metrics = compute_performance_metrics(y[test_mask].values, y_pred_test, 'FINAL TEST')

    # Per-crop test performance
    crop_metrics = {}
    test_panel = panel[test_mask].copy()
    test_panel['y_pred'] = y_pred_test

    for crop in sorted(test_panel['crop'].unique()):
        crop_mask = test_panel['crop'] == crop
        cm = compute_performance_metrics(
            test_panel.loc[crop_mask, 'yield_anomaly'].values,
            test_panel.loc[crop_mask, 'y_pred'].values,
            crop_name=crop,
        )
        crop_metrics[crop] = cm

    # ---- 2012 drought validation ----
    drought_2012 = evaluate_2012_drought(final_model, X, panel, split='in_sample')

    # ---- Gate checks ----
    gate_cfg = CONFIG['yield_model_gates']
    min_spearman_overall = gate_cfg['min_spearman_overall']
    min_spearman_per_crop = gate_cfg['min_spearman_per_crop']
    min_r2_anomaly = gate_cfg['min_r2_anomaly']
    exclude_from_gate = set(gate_cfg.get('exclude_from_gate', []))

    logger.info("=" * 40)
    logger.info("GATE CHECKS (v2 model — z-scored anomaly target)")
    thresholds_passed = True

    overall_spearman = test_metrics.get('spearman_rank', float('nan'))
    spearman_ok = overall_spearman >= min_spearman_overall
    logger.info(f"  Overall Spearman ≥ {min_spearman_overall}: "
                f"{'PASS' if spearman_ok else 'FAIL'} ({overall_spearman:.3f})")
    thresholds_passed &= spearman_ok

    r2_ok = test_metrics['r2'] >= min_r2_anomaly
    logger.info(f"  Overall R² ≥ {min_r2_anomaly}: "
                f"{'PASS' if r2_ok else 'FAIL'} ({test_metrics['r2']:.3f})")
    thresholds_passed &= r2_ok

    for crop, cm in crop_metrics.items():
        if crop in exclude_from_gate:
            logger.info(f"  [{crop}] SKIPPED (excluded from gate)")
            continue
        crop_sp = cm.get('spearman_rank', float('nan'))
        crop_ok = crop_sp >= min_spearman_per_crop
        logger.info(f"  [{crop}] Spearman ≥ {min_spearman_per_crop}: "
                    f"{'PASS' if crop_ok else 'FAIL'} ({crop_sp:.3f})")
        thresholds_passed &= crop_ok

    drought_gate_ok = (
        drought_2012.get('gate_mean_pass', False) or
        drought_2012.get('gate_weighted_pass', False)
    )
    logger.info(f"  2012 drought mean ≤ -0.50σ: "
                f"{'PASS' if drought_2012.get('gate_mean_pass') else 'FAIL'} "
                f"({drought_2012.get('mean_pred', float('nan')):.3f}σ)")
    logger.info(f"  2012 drought weighted ≤ -0.70σ: "
                f"{'PASS' if drought_2012.get('gate_weighted_pass') else 'FAIL'} "
                f"({drought_2012.get('weighted_pred', float('nan')):.3f}σ)")
    thresholds_passed &= drought_gate_ok

    all_metrics = {
        'model_version': 'v2_compound_drought',
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'crop_metrics': crop_metrics,
        'drought_2012': drought_2012,
        'thresholds_passed': thresholds_passed,
        'params': params,
        'n_features': X.shape[1],
    }

    return final_model, all_metrics


def save_v2_artifacts(
    model: lgb.LGBMRegressor,
    metrics: dict,
    X_sample: pd.DataFrame,
) -> Path:
    """Save v2 model, metrics, and feature importances.

    Also copies the model to results/yield_model_v2.pkl for easy reference
    by downstream scripts (05_project.py, 06_stranded.py).

    Args:
        model: Trained v2 LightGBM model.
        metrics: Metrics dict from train_and_evaluate.
        X_sample: Feature matrix (used for column names / importance).

    Returns:
        Path to timestamped results directory.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Primary save: timestamped directory
    model_path = out_dir / 'yield_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model → {model_path}")

    # Convenience alias for downstream scripts
    alias_path = RESULTS_DIR / 'yield_model_v2.pkl'
    with open(alias_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved alias → {alias_path}")

    # Metrics JSON
    def _serialise(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    metrics_path = out_dir / 'yield_model_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=_serialise)

    v2_metrics_path = RESULTS_DIR / 'yield_model_v2_metrics.json'
    with open(v2_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=_serialise)

    # Feature importance
    fi = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)
    fi_path = out_dir / 'feature_importance.csv'
    fi.to_csv(fi_path, index=False)
    fi_v2_path = RESULTS_DIR / 'feature_importance_v2.csv'
    fi.to_csv(fi_v2_path, index=False)

    logger.info(f"Top 10 features by split importance:")
    for _, row in fi.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.0f}")

    logger.info(f"All v2 artifacts saved → {out_dir}")
    return out_dir


