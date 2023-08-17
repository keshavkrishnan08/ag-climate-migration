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

def compute_tail_risk_stranded(
    mean_model: lgb.LGBMRegressor,
    q10_model: lgb.LGBMRegressor,
    panel: pd.DataFrame,
    yield_proj: pd.DataFrame,
    all_crops: list,
    training_columns: list,
    discount_rate: float = 0.04,
    horizon: int = 30,
) -> pd.DataFrame:
    """Compute per-county tail risk premium for stranded asset valuation.

    The mean model tells you expected yield under given climate conditions.
    The Q10 model tells you what yield looks like in a bad year (bottom decile).
    The gap — mean prediction minus Q10 prediction — is the "tail risk premium":
    how much worse things get at the 10th percentile relative to the average.

    Stranded asset tail risk:
        For each county, compute tail_gap = E[yield_anomaly] - Q10[yield_anomaly]
        Convert to $ income via commodity prices and acreage.
        Discount to PV over the projection horizon.
        Sum across counties = additional tail risk stranded value.

    The rationale: farmland buyers pricing tail-risk scenarios should discount
    by the Q10 loss, not just the mean. This component captures that wedge.

    Args:
        mean_model: Trained mean LightGBM model.
        q10_model: Trained Q10 LightGBM model.
        panel: Full feature matrix (for generating predictions on recent baseline).
        yield_proj: Yield projections DataFrame (for acreage and baseline yields).
        all_crops: Full crop category list from training.
        training_columns: Feature column list from training time.
        discount_rate: Real discount rate for PV computation.
        horizon: Projection horizon in years.

    Returns:
        DataFrame with tail risk stranded value per county.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TAIL RISK STRANDED ASSET COMPONENT")
    logger.info("=" * 60)

    # Use recent baseline period (2000-2009) to estimate the current tail gap
    # This represents the climate conditions counties face going into the projection
    baseline = panel[(panel['year'] >= 2000) & (panel['year'] <= TRAIN_END)].copy().reset_index(drop=True)
    logger.info(f"Baseline period 2000-{TRAIN_END}: {len(baseline)} obs across {baseline['fips'].nunique()} counties")

    X_base, _, _ = prepare_features(baseline, all_crops=all_crops, training_columns=training_columns)
    mean_pred = mean_model.predict(X_base)
    q10_pred = q10_model.predict(X_base)
    baseline['mean_pred'] = mean_pred
    baseline['q10_pred'] = q10_pred
    baseline['tail_gap_anomaly'] = baseline['mean_pred'] - baseline['q10_pred']  # > 0 means Q10 is worse

    # County-crop level: average tail gap in anomaly units
    county_crop_gap = (
        baseline.groupby(['fips', 'crop'])
        .agg(
            mean_tail_gap=('tail_gap_anomaly', 'mean'),
            std_yield=('yield_anomaly', 'std'),   # within-county yield volatility
        )
        .reset_index()
    )

    logger.info(f"County-crop tail gaps computed: {len(county_crop_gap)} rows")
    logger.info(f"Mean tail gap (z-scores): {county_crop_gap['mean_tail_gap'].mean():.3f}")
    logger.info(f"P90 tail gap:             {county_crop_gap['mean_tail_gap'].quantile(0.90):.3f}")

    # Merge with projections to get acreage and baseline yield
    proj_base = yield_proj.copy()
    proj_base['price'] = proj_base['crop'].map(COMMODITY_PRICES).fillna(5.0)

    # Recover yield standard deviation from the panel (needed to convert z-score gap → bu/ac)
    # std is crop-specific (yield anomaly is z-scored per county-crop-decade)
    # Use the yield_bu_acre std from baseline for approximation
    baseline_std = (
        baseline.groupby(['fips', 'crop'])['yield_bu_acre']
        .std()
        .reset_index()
        .rename(columns={'yield_bu_acre': 'yield_std_bu'})
    )

    county_crop_gap = county_crop_gap.merge(baseline_std, on=['fips', 'crop'], how='left')
    county_crop_gap['yield_std_bu'] = county_crop_gap['yield_std_bu'].fillna(
        county_crop_gap.groupby('crop')['yield_std_bu'].transform('median')
    )

    # Tail gap in bu/ac = anomaly gap (z-scores) * yield std (bu/ac)
    county_crop_gap['tail_gap_bu'] = county_crop_gap['mean_tail_gap'] * county_crop_gap['yield_std_bu']

    # Merge tail gap onto projections
    proj_base = proj_base.merge(
        county_crop_gap[['fips', 'crop', 'tail_gap_bu', 'mean_tail_gap']],
        on=['fips', 'crop'],
        how='left',
    )
    proj_base['tail_gap_bu'] = proj_base['tail_gap_bu'].fillna(0.0)

    # Discount factors
    min_year = proj_base['year'].min()
    proj_base['years_ahead'] = proj_base['year'] - min_year + 1
    proj_base = proj_base[proj_base['years_ahead'] <= horizon]
    proj_base['discount_factor'] = 1.0 / (1 + discount_rate) ** proj_base['years_ahead']

    # Tail risk income loss per acre per year = tail_gap_bu * price
    # Total income loss = tail_gap income * acres
    proj_base['tail_income_loss'] = (
        proj_base['tail_gap_bu'] * proj_base['price'] * proj_base['acres_harvested']
    )
    proj_base['pv_tail_income'] = proj_base['tail_income_loss'] * proj_base['discount_factor']

    # County-level aggregation across crops and years
    county_tail = (
        proj_base.groupby('fips')
        .agg(
            pv_tail_total=('pv_tail_income', 'sum'),
            total_acres=('acres_harvested', 'mean'),
            mean_tail_gap_anomaly=('mean_tail_gap', 'mean'),
        )
        .reset_index()
    )

    # Tail risk stranded value = PV of the tail gap income
    county_tail['tail_risk_stranded'] = county_tail['pv_tail_total'].clip(lower=0)
    county_tail['tail_risk_per_acre'] = (
        county_tail['tail_risk_stranded'] / county_tail['total_acres'].replace(0, np.nan)
    )

    total_tail_B = county_tail['tail_risk_stranded'].sum() / 1e9
    n_exposed = (county_tail['tail_risk_stranded'] > 0).sum()
    mean_per_acre = county_tail.loc[county_tail['tail_risk_stranded'] > 0, 'tail_risk_per_acre'].mean()

    logger.info(f"\nTail risk stranded asset results (r={discount_rate}, h={horizon}yr):")
    logger.info(f"  Counties with tail risk exposure:  {n_exposed}")
    logger.info(f"  Total tail risk stranded:          ${total_tail_B:.2f}B")
    logger.info(f"  Mean tail risk per acre:           ${mean_per_acre:.0f}/acre")

    # Load current mean-model stranded estimate for comparison
    stranded_path = PROJECT_ROOT / 'results' / 'stranded_assets' / 'stranded_national_SSP245.parquet'
    if stranded_path.exists():
        existing = pd.read_parquet(stranded_path)
        existing_pos = existing[existing['stranded_value_total'] > 0]
        existing_total_B = existing_pos['stranded_value_total'].sum() / 1e9
        logger.info(f"\nComparison to existing mean-model estimate:")
        logger.info(f"  Existing (mean model, ML only):    ${existing_total_B:.1f}B")
        logger.info(f"  Tail risk add-on (Q10 premium):    ${total_tail_B:.2f}B")
        logger.info(f"  Combined (mean + tail risk):       ${existing_total_B + total_tail_B:.1f}B")
        logger.info(f"  Tail risk as % of mean estimate:   {total_tail_B/existing_total_B*100:.1f}%")

    county_tail['discount_rate'] = discount_rate
    county_tail['horizon'] = horizon
    county_tail['alpha'] = ALPHA
    county_tail['method'] = 'Q10_tail_risk_premium'

    return county_tail


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_artifacts(
    model: lgb.LGBMRegressor,
    metrics: dict,
    drought_df: pd.DataFrame,
    tail_risk_df: pd.DataFrame,
) -> None:
    """Save Q10 model, metrics, drought diagnosis, and tail risk estimates.

    Args:
        model: Trained Q10 LightGBM model.
        metrics: Performance metrics dict.
        drought_df: 2012 drought diagnosis DataFrame.
        tail_risk_df: Tail risk stranded asset DataFrame.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = MODELS_DIR / 'yield_model_q10.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Q10 model saved → {model_path}")

    # Metrics
    metrics_path = RESULTS_DIR / 'q10_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    # 2012 drought
    drought_path = RESULTS_DIR / 'drought_2012_q10.csv'
    drought_df.to_csv(drought_path, index=False)
    logger.info(f"2012 drought diagnosis saved → {drought_path}")

    # Tail risk
    tail_path = RESULTS_DIR / 'tail_risk_stranded.parquet'
    tail_risk_df.to_parquet(tail_path, index=False)
    logger.info(f"Tail risk stranded values saved → {tail_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_quantile_model() -> dict:
    """Execute quantile model pipeline end-to-end.

    Returns:
        Dict with model, metrics, drought summary, and tail risk totals.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3C: QUANTILE REGRESSION FOREST (Q10) — TAIL RISK")
    logger.info("=" * 60)

    # Load feature matrix
    panel_path = DATA_PROCESSED / 'feature_matrix.parquet'
    if not panel_path.exists():
        logger.error(f"Feature matrix not found at {panel_path} — run Phase 2 first")
        return {}

    panel = pd.read_parquet(panel_path)
    # Restrict to 1950-2023 (exclude 2024-2025 partial years)
    panel = panel[panel['year'] <= TEST_END].reset_index(drop=True)
    logger.info(f"Loaded feature matrix: {panel.shape} (years {panel['year'].min()}-{panel['year'].max()})")

    # 1. Train Q10 model
    q10_model, metrics, training_columns, all_crops = train_q10_model(panel)

    # 2. 2012 drought diagnosis
    drought_df = diagnose_2012_drought(q10_model, panel, all_crops, training_columns)

    # 3. Train companion MEAN model on the same split and feature set for fair comparison
    logger.info("\n" + "-" * 40)
    logger.info("Training companion MEAN model on same split (train ≤2009)")
    logger.info("-" * 40)

    X_full, y_full, years_full = prepare_features(panel, all_crops=all_crops, training_columns=training_columns)
    train_mask = years_full <= TRAIN_END

    mean_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': CONFIG['yield_model']['n_estimators'],
        'learning_rate': CONFIG['yield_model']['learning_rate'],
        'max_depth': CONFIG['yield_model']['max_depth'],
        'num_leaves': CONFIG['yield_model']['num_leaves'],
        'min_child_samples': CONFIG['yield_model'].get('min_child_samples', 20),
        'subsample': CONFIG['yield_model'].get('subsample', 0.8),
        'colsample_bytree': CONFIG['yield_model'].get('colsample_bytree', 0.8),
        'reg_alpha': CONFIG['yield_model'].get('reg_alpha', 0.1),
        'reg_lambda': CONFIG['yield_model'].get('reg_lambda', 1.0),
        'random_state': RANDOM_SEED,
        'verbose': -1,
    }

    mean_model = lgb.LGBMRegressor(**mean_params)
    mean_model.fit(
        X_full[train_mask.values], y_full[train_mask.values],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    # 2012 mean model prediction for direct comparison
    drought_2012 = panel[(panel['year'] == 2012) & (panel['crop'] == 'corn')].copy()
    X_2012, _, _ = prepare_features(drought_2012, all_crops=all_crops, training_columns=training_columns)
    mean_pred_2012 = mean_model.predict(X_2012)
    logger.info(f"\nMean model 2012 corn prediction:  {mean_pred_2012.mean():.3f} z-scores")
    q10_pred_2012 = q10_model.predict(X_2012)
    logger.info(f"Q10  model 2012 corn prediction:  {q10_pred_2012.mean():.3f} z-scores")
    logger.info(f"Observed 2012 corn mean anomaly:  {drought_2012['yield_anomaly'].mean():.3f} z-scores")

    metrics['mean_model_2012_prediction'] = float(mean_pred_2012.mean())
    metrics['q10_model_2012_prediction'] = float(q10_pred_2012.mean())
    metrics['observed_2012_anomaly'] = float(drought_2012['yield_anomaly'].mean())

    # 4. Tail risk stranded assets
    proj_path = PROJECTIONS_DIR / 'yield_projections_SSP245.parquet'
    if not proj_path.exists():
        logger.warning("Yield projections not found — skipping tail risk computation")
        tail_risk_df = pd.DataFrame()
    else:
        yield_proj = pd.read_parquet(proj_path)
        r = CONFIG['stranded_assets']['discount_rate']
        h = CONFIG['stranded_assets']['projection_horizon']
        tail_risk_df = compute_tail_risk_stranded(
            mean_model, q10_model, panel, yield_proj,
            all_crops=all_crops, training_columns=training_columns,
            discount_rate=r, horizon=h,
        )
        tail_total_B = tail_risk_df['tail_risk_stranded'].sum() / 1e9
        metrics['tail_risk_stranded_B'] = float(tail_total_B)

    # 5. Save everything
    save_artifacts(q10_model, metrics, drought_df, tail_risk_df)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3C SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Q10 model test (2017-2023):")
    logger.info(f"  Pinball loss (Q10):           {metrics['pinball_loss_q10']:.4f}")
    logger.info(f"  Spearman ρ:                   {metrics['spearman_rho']:.3f}")
    logger.info(f"  Coverage (% actual < Q10):    {metrics['coverage_pct']:.1f}%  (target: 10%)")
    logger.info(f"\n2012 drought diagnosis:")
    logger.info(f"  Observed mean corn anomaly:   {metrics['observed_2012_anomaly']:.3f} z-scores")
    logger.info(f"  Mean model prediction:        {metrics['mean_model_2012_prediction']:.3f} z-scores")
    logger.info(f"  Q10 model prediction:         {metrics['q10_model_2012_prediction']:.3f} z-scores")
    if 'tail_risk_stranded_B' in metrics:
        logger.info(f"\nTail risk stranded add-on:")
        logger.info(f"  Q10 tail risk premium:        ${metrics['tail_risk_stranded_B']:.2f}B")
    logger.info("=" * 60)

    return {
        'q10_model': q10_model,
        'mean_model': mean_model,
        'metrics': metrics,
        'drought_2012': drought_df,
        'tail_risk': tail_risk_df,
