"""Phase 4: Future projections — CMIP6 scenarios 2025-2050.

Projects county-level crop yields using:
1. Trained yield model (Phase 3A)
2. Crop switching models (Phase 3B)
3. CMIP6 climate projections downscaled to county level (pre-computed)

Primary scenario: SSP2-4.5 (~RCP 4.5), +1.4-1.8°C by 2050.
GCM ensemble: 5 CMIP6 models, median + 10-90th percentile uncertainty.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
import yaml
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
PROJECTIONS_DIR = PROJECT_ROOT / 'data' / 'projections'
RESULTS_DIR = PROJECT_ROOT / 'results'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

SCENARIOS = CONFIG['climate_scenarios']
RANDOM_SEED = CONFIG['yield_model']['random_seed']


def load_trained_models() -> dict:
    """Load yield model and switching models from Phase 3.

    Returns:
        Dict with 'yield_model' and 'switching_models' keys.

    Raises:
        FileNotFoundError: If model files don't exist.
    """
    results_dirs = sorted(RESULTS_DIR.glob('20*'))
    if not results_dirs:
        raise FileNotFoundError("No results directory found — run Phase 3 first")

    models = {}

    # Find yield model across results dirs
    for d in reversed(results_dirs):
        yield_path = d / 'yield_model.pkl'
        if yield_path.exists():
            with open(yield_path, 'rb') as f:
                models['yield_model'] = pickle.load(f)
            logger.info(f"Loaded yield model from {yield_path}")
            break

    switching_dir = RESULTS_DIR / 'switching_models'
    if switching_dir.exists():
        models['switching_models'] = {}
        for pkl_file in switching_dir.glob('*_model.pkl'):
            pair_name = pkl_file.stem.replace('_model', '')
            with open(pkl_file, 'rb') as f:
                models['switching_models'][pair_name] = pickle.load(f)
        logger.info(f"Loaded {len(models['switching_models'])} switching models")

    return models


def _f_to_c(f_val):
    """Convert Fahrenheit to Celsius."""
    return (f_val - 32) * 5 / 9


def project_yields(
    yield_model,
    climate_proj: pd.DataFrame,
    panel: pd.DataFrame,
    scenario: str
) -> pd.DataFrame:
    """Project county-crop yields under a climate scenario.

    Uses the trained LightGBM yield model on modified feature vectors.
    For each projection year:
      1. Start from most recent observed features per county-crop
      2. Apply climate deltas from CMIP6 projections
      3. Predict yield anomaly with the model
      4. Re-add extrapolated technology trend

    Args:
        yield_model: Trained LGBMRegressor.
        climate_proj: County-year climate projections with delta columns.
        panel: Full feature matrix (training data).
        scenario: Climate scenario name.

    Returns:
        DataFrame with projected yields by county-crop-year.
    """
    logger.info(f"Projecting yields under {scenario}...")

    crops = CONFIG['crops']['primary']

    # Get model feature names
    feature_cols = yield_model.feature_name_

    # Compute per-crop detrended yield std (for converting z-score anomaly → bu/acre)
    crop_detrended_std = {}
    for crop in crops:
        cp = panel[panel['crop'] == crop]
        if cp.empty:
            continue
        detrended = cp['yield_bu_acre'] - (
            cp['yield_trend_intercept'] + cp['yield_trend_slope_15yr'] * cp['year']
        )
        crop_detrended_std[crop] = detrended.std()
    logger.info(f"  Detrended yield std: " +
                ", ".join(f"{c}={v:.1f}" for c, v in sorted(crop_detrended_std.items())))

    # Build baseline: most recent year per county-crop
    max_year = panel['year'].max()
    recent_years = panel[panel['year'] >= max_year - 2]
    baseline = recent_years.groupby(['fips', 'crop'], as_index=False).agg('last')
    logger.info(f"  Baseline: {len(baseline)} county-crop pairs (year={max_year})")

    # One-hot encode crop in baseline
    for c in crops:
        col = f'crop_{c}'
        if col not in baseline.columns:
            baseline[col] = (baseline['crop'] == c).astype(float)

    # Check climate projection columns exist
    has_deltas = 'delta_tmax_july' in climate_proj.columns

    all_projections = []
    projection_years = sorted(climate_proj['year'].unique())

    for year in projection_years:
        year_climate = climate_proj[climate_proj['year'] == year].set_index('fips')
        years_ahead = year - max_year

        for crop in crops:
            crop_base = baseline[baseline['crop'] == crop].copy()
            if crop_base.empty:
                continue

            # Merge with climate projections
            merged = crop_base.set_index('fips')

            # Apply climate deltas to features
            if has_deltas:
                # Get deltas for counties in this crop
                common_fips = merged.index.intersection(year_climate.index)
                if len(common_fips) == 0:
                    continue
                merged = merged.loc[common_fips].copy()
                deltas = year_climate.loc[common_fips]

                # Climate is in °F, model features in °C
                # delta_tmax in °F → ΔC = Δ°F × 5/9
                delta_tmax_c = deltas['delta_tmax_july'] * 5 / 9
                delta_tmax_grow_c = deltas['delta_tmax_growing'] * 5 / 9
                delta_tmin_grow_c = deltas.get('delta_tmin_growing', 0) * 5 / 9
                delta_precip = deltas['delta_precip_growing']

                # Update climate features with deltas
                merged['tmax_july_c'] = merged['tmax_july_c'] + delta_tmax_c
                merged['tmax_growing_c'] = merged['tmax_growing_c'] + delta_tmax_grow_c
                merged['tmin_growing_c'] = merged['tmin_growing_c'] + delta_tmin_grow_c
                merged['precip_growing'] = merged['precip_growing'] + delta_precip

                # Update trend features (warming rate accelerates)
                merged['tmax_july_c_trend10'] = merged['tmax_july_c_trend10'] + delta_tmax_c * 0.1
                merged['tmax_july_c_anomaly'] = delta_tmax_c  # anomaly = departure from baseline

                # CDD increases with warming (rough: +100 CDD per °C warming)
                merged['cdd_annual'] = merged['cdd_annual'] + delta_tmax_c * 100

                # GDD adjustments (warming shifts GDD accumulation)
                for gdd_crop in crops:
                    gdd_col = f'gdd_{gdd_crop}'
                    if gdd_col in merged.columns:
                        merged[gdd_col] = merged[gdd_col] + delta_tmax_grow_c * 50

                # More extreme heat months with warming
                merged['extreme_heat_months'] = merged['extreme_heat_months'] + np.maximum(delta_tmax_c * 0.5, 0)

                # Update precip anomaly (relative to baseline)
                if 'precip_growing_anomaly' in merged.columns:
                    merged['precip_growing_anomaly'] = merged['precip_growing_anomaly'] + delta_precip

                # Update monthly-derived features if v2 model expects them
                # tmax_peak_c increases with July warming
                if 'tmax_peak_c' in merged.columns:
                    merged['tmax_peak_c'] = merged['tmax_peak_c'] + delta_tmax_c
                if 'tmax_peak_c_anomaly' in merged.columns:
                    merged['tmax_peak_c_anomaly'] = merged['tmax_peak_c_anomaly'] + delta_tmax_c
                # precip_jja decreases with precip reduction
                if 'precip_jja' in merged.columns:
                    merged['precip_jja'] = merged['precip_jja'] + delta_precip * 0.5  # JJA is ~half growing season
                if 'precip_jja_anomaly' in merged.columns:
                    merged['precip_jja_anomaly'] = merged['precip_jja_anomaly'] + delta_precip * 0.5
                # PDSI worsens with heat + drying (rough: -0.5 per °C warming)
                if 'pdsi_peak_drought' in merged.columns:
                    merged['pdsi_peak_drought'] = merged['pdsi_peak_drought'] - delta_tmax_c * 0.5
                if 'pdsi_peak_drought_anomaly' in merged.columns:
                    merged['pdsi_peak_drought_anomaly'] = merged['pdsi_peak_drought_anomaly'] - delta_tmax_c * 0.5
                # EDD months increase with warming (~0.3 months per °C in crop belt)
                if 'edd_months_c' in merged.columns:
                    merged['edd_months_c'] = np.maximum(0, merged['edd_months_c'] + delta_tmax_c * 0.3)
                if 'edd_months_c_anomaly' in merged.columns:
                    merged['edd_months_c_anomaly'] = merged['edd_months_c_anomaly'] + delta_tmax_c * 0.3

                # Recalculate compound interaction features from updated climate state
                if 'heat_x_drought' in merged.columns:
                    merged['heat_x_drought'] = (
                        merged['tmax_july_c_anomaly'] * (-merged['pdsi_growing_anomaly'])
                    )
                if 'heat_x_precip' in merged.columns:
                    merged['heat_x_precip'] = (
                        merged['tmax_july_c_anomaly'] * (-merged['precip_growing_anomaly'])
                    )
                if 'extreme_compound' in merged.columns:
                    merged['extreme_compound'] = (
                        (merged['tmax_july_c_anomaly'] > 1.0) &
                        (merged['pdsi_growing_anomaly'] < -1.0)
                    ).astype(float)
                if 'tmax_july_sq' in merged.columns:
                    merged['tmax_july_sq'] = merged['tmax_july_c_anomaly'] ** 2
                if 'edd_x_pdsi' in merged.columns:
                    merged['edd_x_pdsi'] = (
                        merged.get('edd_months_c', 0) * (-merged.get('pdsi_peak_drought', 0))
                    )

            # Extrapolate technology trend
            tech_yield = merged['yield_bu_acre'] + merged['yield_trend_slope_15yr'] * years_ahead
            tech_yield = np.maximum(tech_yield, 0)

            # Prepare features for model prediction
            X = merged.reindex(columns=feature_cols)
            # Fill any missing columns with 0
            X = X.fillna(0)

            # Predict yield anomaly (z-score)
            pred_anomaly = yield_model.predict(X)

            # Also predict baseline anomaly (what model says under current climate)
            X_baseline = crop_base.set_index('fips').loc[common_fips].copy()
            # One-hot encode crop for baseline too
            for c_name in crops:
                col = f'crop_{c_name}'
                if col not in X_baseline.columns:
                    X_baseline[col] = (crop == c_name) * 1.0
            X_base_feat = X_baseline.reindex(columns=feature_cols).fillna(0)
            baseline_anomaly = yield_model.predict(X_base_feat)

            # Climate impact = difference between projected and baseline anomaly
            # This isolates the pure climate-driven shift
            anomaly_delta = pred_anomaly - baseline_anomaly

            # Convert z-score delta to bu/acre using per-crop detrended std
            detrended_std = crop_detrended_std.get(crop, 15.0)
            climate_impact = anomaly_delta * detrended_std

            # Final projected yield = technology trend + climate impact
            yield_projected = tech_yield + climate_impact

            # Uncertainty from GCM spread
            if 'tmax_july_p10' in year_climate.columns and 'tmax_july_p90' in year_climate.columns:
                spread = (year_climate.loc[common_fips, 'tmax_july_p90'] -
                          year_climate.loc[common_fips, 'tmax_july_p10']) * 5 / 9
                uncertainty_pct = np.clip(spread * 0.03, 0.05, 0.25)
            else:
                uncertainty_pct = 0.10

            result_df = pd.DataFrame({
                'fips': common_fips,
                'year': year,
                'crop': crop,
                'scenario': scenario,
                'yield_projected': np.asarray(yield_projected),
                'yield_baseline': np.asarray(merged['yield_bu_acre']),
                'yield_tech_trend': np.asarray(tech_yield),
                'climate_impact_bu': np.asarray(climate_impact),
                'yield_p10': np.asarray(yield_projected * (1 - uncertainty_pct)),
                'yield_p90': np.asarray(yield_projected * (1 + uncertainty_pct)),
                'acres_harvested': np.asarray(merged['acres_harvested']),
            })
            all_projections.append(result_df)

    result = pd.concat(all_projections, ignore_index=True) if all_projections else pd.DataFrame()
    logger.info(f"  {scenario}: {len(result)} county-crop-year projections "
                f"({result['fips'].nunique() if not result.empty else 0} counties)")
    return result


def project_switching(
    switching_models: dict,
    climate_proj: pd.DataFrame,
    panel: pd.DataFrame,
    scenario: str
) -> pd.DataFrame:
    """Project crop switching probabilities under a climate scenario.

    Args:
        switching_models: Dict of trained switching models.
        climate_proj: Projected climate data.
        panel: Feature matrix.
        scenario: Climate scenario name.

    Returns:
        DataFrame with switching probabilities by county-pair-year.
    """
    logger.info(f"Projecting crop switching under {scenario}...")

    crops = CONFIG['crops']['primary']
    max_year = panel['year'].max()
    recent = panel[panel['year'] >= max_year - 2]
    baseline = recent.groupby(['fips', 'crop'], as_index=False).agg('last')

    # One-hot encode crop
    for c in crops:
        col = f'crop_{c}'
        if col not in baseline.columns:
            baseline[col] = (baseline['crop'] == c).astype(float)

    all_projections = []
    projection_years = sorted(climate_proj['year'].unique())
    # Sample every 5 years for switching (it's slow)
    sample_years = [y for y in projection_years if y % 5 == 0 or y == projection_years[-1]]

    for pair_name, model in switching_models.items():
        if model is None:
            continue

        from_crop, to_crop = pair_name.split('_to_')
        crop_data = baseline[baseline['crop'] == from_crop].copy()
        if crop_data.empty:
            continue

        # Get features the switching model expects (28 features, no crop dummies)
        exclude = {'fips', 'year', 'crop', 'yield_bu_acre', 'yield_anomaly',
                    'acres_harvested', 'production', 'switched'}
        # Also exclude crop dummies — switching models were trained without them
        exclude.update({f'crop_{c}' for c in crops})
        feature_cols = [c for c in crop_data.columns
                        if c not in exclude and crop_data[c].dtype in ('float64', 'float32', 'int64')]

        for year in sample_years:
            year_climate = climate_proj[climate_proj['year'] == year].set_index('fips')

            features = crop_data.set_index('fips').copy()
            common_fips = features.index.intersection(year_climate.index)
            if len(common_fips) == 0:
                continue
            features = features.loc[common_fips]

            # Apply warming deltas
            if 'delta_tmax_july' in year_climate.columns:
                delta_c = year_climate.loc[common_fips, 'delta_tmax_july'] * 5 / 9
                features['tmax_july_c'] = features['tmax_july_c'] + delta_c
                features['tmax_july_c_trend10'] = features['tmax_july_c_trend10'] + delta_c * 0.1

            X = features[feature_cols].fillna(0)

            try:
                probs = model.predict_proba(X)[:, 1]
            except Exception:
                probs = np.zeros(len(X))

            result_df = pd.DataFrame({
                'fips': common_fips,
                'year': year,
                'from_crop': from_crop,
                'to_crop': to_crop,
                'scenario': scenario,
                'switch_probability': probs,
            })
            all_projections.append(result_df)

    result = pd.concat(all_projections, ignore_index=True) if all_projections else pd.DataFrame()
    logger.info(f"  {scenario}: {len(result)} switching projections")
    return result


def validate_hindcast(
    yield_model,
    panel: pd.DataFrame,
    test_start: int = 2013,
    test_end: int = 2023
) -> dict:
    """Validate model by hindcasting 2013-2023 period.

    Args:
        yield_model: Trained yield model.
        panel: Feature matrix including test period.
        test_start: Start of hindcast period.
        test_end: End of hindcast period.

    Returns:
        Dict of hindcast validation metrics.
    """
    logger.info("=" * 40)
    logger.info("HINDCAST VALIDATION 2013-2023")
    logger.info("=" * 40)

    crops_list = CONFIG['crops']['primary']
    test = panel[(panel['year'] >= test_start) & (panel['year'] <= test_end)].copy()

    # One-hot encode crop
    for c in crops_list:
        col = f'crop_{c}'
        if col not in test.columns:
            test[col] = (test['crop'] == c).astype(float)

    feature_cols = yield_model.feature_name_

    results = {}
    for crop in test['crop'].unique():
        crop_data = test[test['crop'] == crop].copy()
        if len(crop_data) < 10:
            continue

        X = crop_data.reindex(columns=feature_cols).fillna(0)
        y_true = crop_data['yield_anomaly'].values

        y_pred = yield_model.predict(X)

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        spearman_r, _ = stats.spearmanr(y_true, y_pred)

        results[crop] = {
            'n_obs': len(crop_data),
            'rmse': float(rmse),
            'spearman_rank': float(spearman_r),
        }
        logger.info(f"  {crop:15s}  RMSE={rmse:.3f}  Spearman={spearman_r:.3f}  n={len(crop_data)}")

    # 2012 drought test
    drought_2012 = panel[(panel['year'] == 2012) & (panel['crop'] == 'corn')].copy()
    if not drought_2012.empty:
        # Add crop dummies BEFORE reindex so they get correct values (not filled with 0)
        for c in crops_list:
            drought_2012[f'crop_{c}'] = (drought_2012['crop'] == c).astype(float)
        X_drought = drought_2012.reindex(columns=feature_cols).fillna(0)
        drought_pred = yield_model.predict(X_drought)
        acres_2012 = drought_2012['acres_harvested'].fillna(1).values
        import numpy as _np
        weighted_pred = float(_np.average(drought_pred, weights=acres_2012))
        logger.info(f"  2012 drought: mean={drought_pred.mean():.3f}σ, "
                    f"acreage-weighted={weighted_pred:.3f}σ "
                    f"(observed: {drought_2012['yield_anomaly'].mean():.3f}σ)")
        results['drought_2012'] = {
            'mean_predicted_anomaly': float(drought_pred.mean()),
            'weighted_predicted_anomaly': weighted_pred,
            'observed_anomaly': float(drought_2012['yield_anomaly'].mean()),
            'passed': drought_pred.mean() < -0.5,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_projections() -> dict:
    """Execute full projection pipeline under all scenarios.

    Returns:
        Dict with yield and switching projections for all scenarios.
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: CLIMATE PROJECTIONS 2025-2050")
    logger.info("=" * 60)

    # Load models
    try:
        models = load_trained_models()
    except FileNotFoundError as e:
        logger.error(str(e))
        return {}

    if 'yield_model' not in models:
        logger.error("No yield model found — run Phase 3 first")
        return {}

    # Load feature matrix
    panel_path = DATA_PROCESSED / 'feature_matrix.parquet'
    panel = pd.read_parquet(panel_path)
    logger.info(f"Loaded panel: {len(panel)} rows, {panel['fips'].nunique()} counties")

    # If the yield model expects v2 interaction features, add them to the panel.
    # This is safe: if the model doesn't use a feature, LightGBM ignores it.
    yield_model = models['yield_model']
    model_features = set(yield_model.feature_name_) if hasattr(yield_model, 'feature_name_') else set()
    if 'heat_x_drought' in model_features:
        logger.info("v2 model detected — enriching panel with compound drought features")
        panel['fips'] = panel['fips'].astype(str)

        # Monthly climate features for compound drought interactions
        monthly_path = DATA_RAW / 'prism' / 'county_climate_monthly.parquet'
        if monthly_path.exists():
            monthly = pd.read_parquet(monthly_path)
            for m in ['05', '06', '07', '08']:
                monthly[f'tmax_m{m}_c'] = (monthly[f'tmax_m{m}'] - 32) * 5 / 9
            monthly['tmax_peak_c'] = monthly[['tmax_m06_c', 'tmax_m07_c', 'tmax_m08_c']].max(axis=1)
            monthly['precip_jja'] = monthly[['precip_m06', 'precip_m07', 'precip_m08']].sum(axis=1)
            monthly['pdsi_peak_drought'] = monthly[['pdsi_m06', 'pdsi_m07', 'pdsi_m08']].min(axis=1)
            monthly['edd_months_c'] = (
