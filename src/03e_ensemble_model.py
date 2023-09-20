"""Phase 3E: Ensemble yield model (LightGBM v2 + Ridge + RandomForest).

The base LightGBM v2 model achieves R²=0.21 on z-scored yield anomalies.
Ensembles routinely add 0.03-0.08 R² by averaging diverse predictors that
each capture different aspects of the yield-climate relationship:
  - LightGBM v2: deep tree, compound drought interactions, threshold effects
  - Ridge:        linear climate signal, regularised against noise
  - RandomForest: bagged trees, robust to outlier years

All three train on IDENTICAL features and splits. The ensemble is a simple
unweighted average (equal weights).

Approaches tested:
  1. Ensemble (LightGBM + Ridge + RF), train ≤ 2012, test 2013-2023
  2. Extended training set (train ≤ 2012, adds 2010-2012 vs prior ≤ 2009 split)
     — already the default in v2; carried forward here.

Temporal splits (match CLAUDE.md Critical Rules):
  Train:  years ≤ 2012 (val_end in config.yaml)
  Test:   2013-2023

Gate targets:
  R² ≥ 0.25, Spearman ≥ 0.50

Saves (if ensemble beats v2 alone):
  results/yield_model_ensemble.pkl
  results/yield_model_ensemble_metrics.json

Usage:
  python src/03e_ensemble_model.py
"""

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml
from loguru import logger
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.validation import (
    check_no_future_leakage,
    compute_performance_metrics,
    temporal_rolling_cv,
)

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

with open(PROJECT_ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

RANDOM_SEED = CONFIG["yield_model"]["random_seed"]
VAL_END = CONFIG["temporal"]["val_end"]      # 2012
TEST_END = CONFIG["temporal"]["test_end"]    # 2023

# Schlenker & Roberts EDD threshold
EDD_THRESHOLD_C = 33.5
PEAK_MONTHS = ["06", "07", "08"]


# ---------------------------------------------------------------------------
# Feature engineering (mirrors 03d_retrain_with_interactions.py exactly)
# ---------------------------------------------------------------------------

def load_monthly_features() -> pd.DataFrame:
    """Load monthly climate data and compute peak-season features.

    Converts nClimDiv temperatures (°F) to Celsius, then derives:
      - tmax_peak_c:       max(Jun,Jul,Aug) Tmax
      - precip_jja:        sum of Jun+Jul+Aug precipitation
      - pdsi_peak_drought: min(Jun,Jul,Aug PDSI) — worst drought month
      - edd_months_c:      count of months above 33.5°C EDD threshold

    Returns:
        DataFrame with fips, year, and four derived columns.
    """
    monthly_path = DATA_RAW / "prism" / "county_climate_monthly.parquet"
    monthly = pd.read_parquet(monthly_path)

    for m in PEAK_MONTHS:
        monthly[f"tmax_m{m}_c"] = (monthly[f"tmax_m{m}"] - 32) * 5 / 9

    tmax_cols = [f"tmax_m{m}_c" for m in PEAK_MONTHS]
    precip_cols = [f"precip_m{m}" for m in PEAK_MONTHS]
    pdsi_cols = [f"pdsi_m{m}" for m in PEAK_MONTHS]

    monthly["tmax_peak_c"] = monthly[tmax_cols].max(axis=1)
    monthly["precip_jja"] = monthly[precip_cols].sum(axis=1)
    monthly["pdsi_peak_drought"] = monthly[pdsi_cols].min(axis=1)
    monthly["edd_months_c"] = (monthly[tmax_cols] > EDD_THRESHOLD_C).sum(axis=1).astype(float)

    out = monthly[["fips", "year", "tmax_peak_c", "precip_jja",
                   "pdsi_peak_drought", "edd_months_c"]].copy()
    out["fips"] = out["fips"].astype(str)
    logger.info(f"Monthly features loaded: {out.shape}")
    return out


def add_interaction_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add compound drought interaction and quadratic features.

    Features added:
      heat_x_drought   — hot AND dry anomaly product
      heat_x_precip    — hot AND low precip anomaly product
      extreme_compound — binary: heat > +1σ AND PDSI < -1σ
      tmax_july_sq     — quadratic heat for yield-cliff threshold
      precip_deficit   — county-crop shortfall vs historical mean

    Args:
        panel: Feature matrix with anomaly columns.

    Returns:
        Panel with five new columns.
    """
    panel = panel.copy()
    panel["heat_x_drought"] = panel["tmax_july_c_anomaly"] * (-panel["pdsi_growing_anomaly"])
    panel["heat_x_precip"] = panel["tmax_july_c_anomaly"] * (-panel["precip_growing_anomaly"])
    panel["extreme_compound"] = (
        (panel["tmax_july_c_anomaly"] > 1.0) & (panel["pdsi_growing_anomaly"] < -1.0)
    ).astype(float)
    panel["tmax_july_sq"] = panel["tmax_july_c_anomaly"] ** 2
    precip_baseline = panel.groupby(["fips", "crop"])["precip_growing"].transform("mean")
    panel["precip_deficit"] = (precip_baseline - panel["precip_growing"]).clip(lower=0)
    logger.info("Added 5 interaction features")
    return panel


def add_monthly_anomaly_features(
    panel: pd.DataFrame,
    monthly_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge monthly features into panel and compute county anomalies.

    Args:
        panel: Feature matrix with fips, year columns.
        monthly_features: Output of load_monthly_features().

    Returns:
        Panel with monthly features and county-level anomaly columns.

    Raises:
        ValueError: If merge changes row count.
    """
    n_before = len(panel)
    panel = panel.merge(monthly_features, on=["fips", "year"], how="left")
    if len(panel) != n_before:
        raise ValueError(f"Merge changed row count: {n_before} → {len(panel)}")

    base_cols = ["tmax_peak_c", "precip_jja", "pdsi_peak_drought", "edd_months_c"]
    for col in base_cols:
        county_mean = panel.groupby("fips")[col].transform("mean")
        panel[f"{col}_anomaly"] = panel[col] - county_mean

    panel["edd_x_pdsi"] = panel["edd_months_c"] * (-panel["pdsi_peak_drought"])
    logger.info(f"Merged monthly features. Panel: {panel.shape}")
    return panel


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract numeric feature columns, excluding identifiers and target.

    Args:
        df: Feature matrix DataFrame.

    Returns:
        List of numeric column names for model input.
    """
    exclude = {
        "fips", "year", "crop",
        "yield_bu_acre", "yield_anomaly",
        "acres_harvested", "production",
    }
    return [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in ("float64", "float32", "int64", "int32", "bool")
        and not df[c].isna().all()
    ]


def prepare_features(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build final feature matrix with crop dummies and zero-fill NAs.

    Args:
        panel: Panel with all features added.

    Returns:
        Tuple of (X, y, years) ready for model training.
    """
    feature_cols = get_feature_columns(panel)
    logger.info(f"Feature count: {len(feature_cols)}")

    X = panel[feature_cols].fillna(0).copy()
    crop_dummies = pd.get_dummies(panel["crop"], prefix="crop")
    X = pd.concat([X, crop_dummies], axis=1)

    y = panel["yield_anomaly"].copy()
    years = panel["year"].copy()

    logger.info(f"X={X.shape}, y={y.shape}")
    return X, y, years


# ---------------------------------------------------------------------------
# Individual models
# ---------------------------------------------------------------------------

def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.LGBMRegressor:
    """Train LightGBM v2 with compound drought hyperparameters.

    Matches 03d exactly: depth=8, leaves=127, lr=0.02, n_est=1500.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features for early stopping evaluation.
        y_val: Validation targets.

    Returns:
        Fitted LGBMRegressor.
    """
    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 1500,
        "learning_rate": 0.02,
        "max_depth": 8,
        "num_leaves": 127,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "random_state": RANDOM_SEED,
        "verbose": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return model


def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Ridge, StandardScaler]:
    """Train Ridge regression with StandardScaler normalisation.

    Ridge captures the linear component of the climate-yield relationship.
    Alpha=10 provides moderate regularisation on ~50 features.

    Args:
        X_train: Training features.
        y_train: Training targets.

    Returns:
        Tuple of (fitted Ridge, fitted StandardScaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=10.0, random_state=RANDOM_SEED)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestRegressor:
    """Train RandomForest with conservative depth to avoid overfitting.

    n_estimators=150, max_depth=10, max_samples=0.3 keeps training fast
    on 582K observations (roughly 175K samples per tree). min_samples_leaf=30
    prevents individual year/county quirks from dominating predictions.

    Args:
        X_train: Training features.
        y_train: Training targets.

    Returns:
        Fitted RandomForestRegressor.
    """
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=30,
        max_features=0.4,
        max_samples=0.3,      # subsample 30% per tree — huge speedup, minimal quality loss
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Ensemble prediction
# ---------------------------------------------------------------------------

def fit_blend_weights(
    lgbm_model: lgb.LGBMRegressor,
    ridge_model: Ridge,
    rf_model: RandomForestRegressor,
    scaler: StandardScaler,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> np.ndarray:
    """Fit non-negative blend weights by minimising RMSE on a validation set.

    Rather than equal weights (which dilutes LightGBM's superior signal with
    weaker Ridge/RF predictions), we solve for the optimal convex combination:
        w* = argmin ||y_val - (w1*p_lgbm + w2*p_ridge + w3*p_rf)||²
        subject to w >= 0, sum(w) = 1

    Uses NNLS (non-negative least squares) followed by L1 normalisation.

    Args:
        lgbm_model: Trained LightGBM.
        ridge_model: Trained Ridge regression.
        rf_model: Trained RandomForest.
        scaler: StandardScaler for Ridge input.
        X_val: Held-out validation features.
        y_val: Held-out validation targets.

    Returns:
        Array of three blend weights [w_lgbm, w_ridge, w_rf].
    """
    from scipy.optimize import nnls

    pred_lgbm = lgbm_model.predict(X_val)
    pred_ridge = ridge_model.predict(scaler.transform(X_val))
    pred_rf = rf_model.predict(X_val)

    # Stack predictions as columns: (n, 3)
    P = np.column_stack([pred_lgbm, pred_ridge, pred_rf])
    weights, _ = nnls(P, y_val.values)

    # Normalise to sum to 1
    total = weights.sum()
    if total < 1e-10:
        weights = np.array([1.0, 0.0, 0.0])  # fallback: use lgbm only
    else:
        weights = weights / total

    logger.info(f"Blend weights — LightGBM: {weights[0]:.3f}, Ridge: {weights[1]:.3f}, RF: {weights[2]:.3f}")
    return weights

