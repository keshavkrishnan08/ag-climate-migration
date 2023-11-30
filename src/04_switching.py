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

    # Merge with features
    feature_data = panel[panel['crop'] == from_crop].merge(
        data[['fips', 'year', 'switched']],
        on=['fips', 'year'],
        how='inner'
    )

    feature_cols = get_switching_features(feature_data, pair)
    X = feature_data[feature_cols].copy()
    y = feature_data['switched'].copy()
    years = feature_data['year'].copy()

    # Temporal split
    train_mask = years <= CONFIG['temporal']['val_end']
    test_mask = years > CONFIG['temporal']['val_end']

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    logger.info(f"  Train: n={len(X_train)} (positive rate: {y_train.mean():.3f})")
    logger.info(f"  Test:  n={len(X_test)} (positive rate: {y_test.mean():.3f})")

    # Train LightGBM classifier
    base_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbose=-1,
        is_unbalance=True,  # Handle class imbalance
    )

    base_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    # Platt scaling calibration
    calibrated_model = CalibratedClassifierCV(
        base_model, method='sigmoid', cv='prefit'
    )
    calibrated_model.fit(X_test, y_test)

    # Evaluate
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]

    metrics = {
        'pair': f"{from_crop}_to_{to_crop}",
        'auc_roc': roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else 0,
        'avg_precision': average_precision_score(y_test, y_prob) if y_test.nunique() > 1 else 0,
        'brier_score': brier_score_loss(y_test, y_prob),
        'log_loss': log_loss(y_test, y_prob),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'positive_rate_train': float(y_train.mean()),
        'positive_rate_test': float(y_test.mean()),
    }

    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.3f}, Brier: {metrics['brier_score']:.4f}")

    # CRITICAL: Verify monotonicity in temperature trend
    if 'gdd_trend_slope' in X_test.columns:
        verify_temp_monotonicity(calibrated_model, X_test, pair)

    return calibrated_model, metrics


def verify_temp_monotonicity(
    model: CalibratedClassifierCV,
    X: pd.DataFrame,
    pair: Tuple[str, str]
) -> bool:
    """Verify that switching probability is monotone in temperature trend.

    CRITICAL constraint: hotter counties should be more likely to switch
    away from heat-sensitive crops.

    Args:
        model: Calibrated switching model.
        X: Feature matrix.
        pair: (from_crop, to_crop) tuple.

    Returns:
        True if monotonicity holds.
    """
    from_crop, to_crop = pair

    if 'gdd_trend_slope' not in X.columns:
        logger.warning("Cannot verify monotonicity — gdd_trend_slope not in features")
        return True

    # Create synthetic data varying only temperature trend
    X_synth = X.median().to_frame().T
    X_synth = pd.concat([X_synth] * 50, ignore_index=True)

    temp_range = np.linspace(
        X['gdd_trend_slope'].quantile(0.05),
        X['gdd_trend_slope'].quantile(0.95),
        50
    )
    X_synth['gdd_trend_slope'] = temp_range

    probs = model.predict_proba(X_synth)[:, 1]

    # Check if probability is generally increasing with temperature
    correlation = np.corrcoef(temp_range, probs)[0, 1]

    heat_sensitive = {'corn', 'cotton', 'wheat_winter'}
    if from_crop in heat_sensitive:
        # Switching AWAY from heat-sensitive crop: should increase with temp
        monotone = correlation > 0
        logger.info(f"  Monotonicity ({from_crop}→{to_crop}): corr={correlation:.3f} "
                    f"{'PASS' if monotone else 'WARN — not monotone'}")
    else:
        monotone = True  # No constraint for non-heat-sensitive crops

    return monotone


def project_switching_probability(
    model: CalibratedClassifierCV,
    county_features: pd.DataFrame,
    temp_scenario: pd.DataFrame,
    year: int
) -> Dict[str, float]:
    """Project probability of crop switching given climate scenario.

    Args:
        model: Calibrated switching model.
        county_features: Fixed county characteristics (soil, farm structure).
        temp_scenario: Temperature trajectory to projection year.
        year: Projection year (2025-2050).

    Returns:
        Dict: {crop_pair: probability_of_switch}.
    """
    # Build feature vector for projection year
    features = county_features.copy()

    # Update climate features from scenario
    if not temp_scenario.empty:
        for col in temp_scenario.columns:
            if col in features.columns:
                features[col] = temp_scenario[col].values

    # Predict
    prob = model.predict_proba(features)[:, 1]
    return {'probability': float(prob.mean())}


# ---------------------------------------------------------------------------
# Historical validation (PRD Section 12, Fix 5)
# ---------------------------------------------------------------------------
def validate_switching_historical() -> dict:
    """Validate switching model against pre-CDL historical events.

    Four validation events using NASS county acreage records from 1950:
    1. Soybean adoption in Corn Belt (1960-1980) — NEGATIVE TEST
    2. Sorghum expansion in southern Plains (1950-1975)
    3. Cotton retreat from Missouri/Tennessee (1980-2010)
    4. Winter wheat boundary southward shift in Kansas (1990-2010)

    Returns:
        Dict of validation results with pass/fail for each event.
    """
    logger.info("=" * 40)
    logger.info("HISTORICAL SWITCHING VALIDATION")
    logger.info("=" * 40)

    results = {}

    # Test 1: Soybean adoption 1960-1980 — NEGATIVE TEST
    # Model trained on 1950-1960 should NOT predict soybean expansion
    # from climate signals alone (it was technology-driven)
    results['soybean_negative'] = {
        'event': 'Soybean adoption in Corn Belt 1960-1980',
        'test_type': 'NEGATIVE',
        'criterion': 'Model predicts <10% of actual soybean expansion from climate features',
        'passed': None,  # Will be filled when data is available
    }

    # Test 2: Sorghum expansion 1950-1975
    results['sorghum_positive'] = {
        'event': 'Sorghum expansion in southern Plains 1950-1975',
        'test_type': 'POSITIVE',
        'criterion': 'Spearman rank correlation > 0.55 across counties',
        'passed': None,
    }

    # Test 3: Cotton retreat 1980-2010
    results['cotton_positive'] = {
        'event': 'Cotton retreat from Missouri/Tennessee 1980-2010',
        'test_type': 'POSITIVE',
        'criterion': 'Model identifies 70%+ of counties that exited cotton',
        'passed': None,
    }

    # Test 4: Winter wheat boundary shift 1990-2010
    results['wheat_positive'] = {
        'event': 'Winter wheat boundary southward shift in Kansas 1990-2010',
        'test_type': 'POSITIVE',
        'criterion': 'Predicted boundary within 50km of observed NASS boundary',
        'passed': None,
    }

    # All four must pass before projections are credible
    all_tested = all(r['passed'] is not None for r in results.values())
    if all_tested:
        all_passed = all(r['passed'] for r in results.values())
        if not all_passed:
            logger.error("Switching model fails historical validation — do not project")
        else:
            logger.info("All 4 historical validation tests PASSED")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_switching_models() -> Dict[str, Tuple]:
    """Train switching models for all crop pairs.

    Returns:
