"""Run yield projections for SSP3-7.0 using the existing v2 yield model.

Loads the SSP370 county climate projections and applies the trained LightGBM
yield model, saving results to data/projections/yield_projections_SSP370.parquet.
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
PROJECTIONS_DIR = PROJECT_ROOT / 'data' / 'projections'
RESULTS_DIR = PROJECT_ROOT / 'results'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

SCENARIO = 'SSP370'


def load_yield_model():
    """Load the most recent yield model from results directories.

    Returns:
        Trained LGBMRegressor.

    Raises:
        FileNotFoundError: If no yield model exists.
    """
    results_dirs = sorted(RESULTS_DIR.glob('20*'))
    for d in reversed(results_dirs):
        yield_path = d / 'yield_model.pkl'
        if yield_path.exists():
            with open(yield_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded yield model from {yield_path}")
            return model
    raise FileNotFoundError("No yield model found — run Phase 3 first")


def project_yields_ssp370(yield_model, climate_proj, panel):
    """Project county-crop yields under SSP3-7.0.

    Applies climate deltas from SSP370 projections to the trained yield model.
    Logic mirrors src/05_project.py::project_yields() exactly.

    Args:
        yield_model: Trained LGBMRegressor from Phase 3.
        climate_proj: SSP370 county climate projections DataFrame.
        panel: Feature matrix (training data with all engineered features).

    Returns:
        DataFrame with projected yields by county-crop-year.
    """
    logger.info(f"Projecting yields under {SCENARIO}...")

    crops = CONFIG['crops']['primary']
    feature_cols = yield_model.feature_name_

    # Per-crop detrended yield std for z-score → bu/acre conversion
    crop_detrended_std = {}
    for crop in crops:
        cp = panel[panel['crop'] == crop]
        if cp.empty:
            continue
        detrended = cp['yield_bu_acre'] - (
            cp['yield_trend_intercept'] + cp['yield_trend_slope_15yr'] * cp['year']
        )
        crop_detrended_std[crop] = detrended.std()
    logger.info("  Detrended yield std: " +
                ", ".join(f"{c}={v:.1f}" for c, v in sorted(crop_detrended_std.items())))

    # Baseline: most recent 3 years per county-crop
    max_year = panel['year'].max()
    recent_years = panel[panel['year'] >= max_year - 2]
    baseline = recent_years.groupby(['fips', 'crop'], as_index=False).agg('last')
    logger.info(f"  Baseline: {len(baseline)} county-crop pairs (year={max_year})")

    for c in crops:
        col = f'crop_{c}'
        if col not in baseline.columns:
            baseline[col] = (baseline['crop'] == c).astype(float)

    all_projections = []
    projection_years = sorted(climate_proj['year'].unique())

    for year in projection_years:
        year_climate = climate_proj[climate_proj['year'] == year].set_index('fips')
        years_ahead = year - max_year

        for crop in crops:
            crop_base = baseline[baseline['crop'] == crop].copy()
            if crop_base.empty:
                continue

            merged = crop_base.set_index('fips')
            common_fips = merged.index.intersection(year_climate.index)
            if len(common_fips) == 0:
                continue
            merged = merged.loc[common_fips].copy()
            deltas = year_climate.loc[common_fips]

            # Climate deltas °F → ΔC for model (trained in °C)
            delta_tmax_c      = deltas['delta_tmax_july'] * 5 / 9
            delta_tmax_grow_c = deltas['delta_tmax_growing'] * 5 / 9
            delta_tmin_grow_c = deltas.get('delta_tmin_growing', 0) * 5 / 9
            delta_precip      = deltas['delta_precip_growing']

            merged['tmax_july_c']    = merged['tmax_july_c']    + delta_tmax_c
            merged['tmax_growing_c'] = merged['tmax_growing_c'] + delta_tmax_grow_c
            merged['tmin_growing_c'] = merged['tmin_growing_c'] + delta_tmin_grow_c
            merged['precip_growing'] = merged['precip_growing'] + delta_precip
            merged['tmax_july_c_trend10'] = merged['tmax_july_c_trend10'] + delta_tmax_c * 0.1
            merged['tmax_july_c_anomaly'] = delta_tmax_c
            merged['cdd_annual'] = merged['cdd_annual'] + delta_tmax_c * 100

            for gdd_crop in crops:
                gdd_col = f'gdd_{gdd_crop}'
                if gdd_col in merged.columns:
                    merged[gdd_col] = merged[gdd_col] + delta_tmax_grow_c * 50

            merged['extreme_heat_months'] = merged['extreme_heat_months'] + np.maximum(delta_tmax_c * 0.5, 0)

            if 'precip_growing_anomaly' in merged.columns:
                merged['precip_growing_anomaly'] = merged['precip_growing_anomaly'] + delta_precip
            if 'tmax_peak_c' in merged.columns:
                merged['tmax_peak_c'] = merged['tmax_peak_c'] + delta_tmax_c
            if 'tmax_peak_c_anomaly' in merged.columns:
                merged['tmax_peak_c_anomaly'] = merged['tmax_peak_c_anomaly'] + delta_tmax_c
            if 'precip_jja' in merged.columns:
                merged['precip_jja'] = merged['precip_jja'] + delta_precip * 0.5
            if 'precip_jja_anomaly' in merged.columns:
                merged['precip_jja_anomaly'] = merged['precip_jja_anomaly'] + delta_precip * 0.5
            if 'pdsi_peak_drought' in merged.columns:
                merged['pdsi_peak_drought'] = merged['pdsi_peak_drought'] - delta_tmax_c * 0.5
            if 'pdsi_peak_drought_anomaly' in merged.columns:
                merged['pdsi_peak_drought_anomaly'] = merged['pdsi_peak_drought_anomaly'] - delta_tmax_c * 0.5
            if 'edd_months_c' in merged.columns:
                merged['edd_months_c'] = np.maximum(0, merged['edd_months_c'] + delta_tmax_c * 0.3)
            if 'edd_months_c_anomaly' in merged.columns:
                merged['edd_months_c_anomaly'] = merged['edd_months_c_anomaly'] + delta_tmax_c * 0.3
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

            # Technology trend
            tech_yield = merged['yield_bu_acre'] + merged['yield_trend_slope_15yr'] * years_ahead
            tech_yield = np.maximum(tech_yield, 0)

            X = merged.reindex(columns=feature_cols).fillna(0)
            pred_anomaly = yield_model.predict(X)

            # Baseline anomaly (no climate change)
            X_baseline = crop_base.set_index('fips').loc[common_fips].copy()
            for c_name in crops:
                col = f'crop_{c_name}'
                if col not in X_baseline.columns:
                    X_baseline[col] = (crop == c_name) * 1.0
            X_base_feat = X_baseline.reindex(columns=feature_cols).fillna(0)
            baseline_anomaly = yield_model.predict(X_base_feat)

            anomaly_delta = pred_anomaly - baseline_anomaly
            detrended_std = crop_detrended_std.get(crop, 15.0)
            climate_impact = anomaly_delta * detrended_std
            yield_projected = tech_yield + climate_impact

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
                'scenario': SCENARIO,
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
    logger.info(f"  {SCENARIO}: {len(result)} county-crop-year projections "
                f"({result['fips'].nunique() if not result.empty else 0} counties)")
    return result


def main():
    """Execute SSP370 yield projection pipeline."""
    logger.info("=" * 60)
    logger.info("SSP370 YIELD PROJECTIONS")
    logger.info("=" * 60)

    # Load yield model
    yield_model = load_yield_model()

    # Load feature matrix
    panel_path = DATA_PROCESSED / 'feature_matrix.parquet'
    panel = pd.read_parquet(panel_path)
