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
