"""Compute stranded agricultural assets under SSP3-7.0.

Runs the same conservative (ML only) and central (ML + SR + indirect) methods
as 06_stranded.py but using SSP370 yield and climate projections.
Reports results and compares against the SSP245 baseline.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
PROJECTIONS_DIR = PROJECT_ROOT / 'data' / 'projections'
RESULTS_DIR = PROJECT_ROOT / 'results'
OUTPUT_DIR = RESULTS_DIR / 'stranded_assets'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

SCENARIO = 'SSP370'

# ── Inline helper functions (mirrors 06_stranded.py) ─────────────────────────

COMMODITY_PRICES = {
    'corn': 5.50, 'soybeans': 12.80, 'wheat_winter': 7.20,
    'wheat_spring': 8.10, 'cotton': 0.78, 'sorghum': 5.30,
    'barley': 6.10, 'oats': 3.80,
}

SR_THRESHOLD_MODERATE = 29.0
SR_JULY_DAYS = 31
SR_SHOULDER_DAYS = 60
SR_COEFFICIENTS = {
    'corn':         -0.0662,
    'soybeans':     -0.0560,
    'wheat_winter': -0.0420,
    'wheat_spring': -0.0420,
    'cotton':       -0.0662,
    'sorghum':      -0.0662,
    'barley':       -0.0420,
    'oats':         -0.0420,
}


def compute_stranded_vectorized(
    yield_proj: pd.DataFrame,
    land_values: pd.DataFrame,
    discount_rate: float = 0.04,
    horizon: int = 30,
    scenario: str = 'SSP370'
) -> pd.DataFrame:
    """Compute stranded assets (ML model only) across all counties/crops.

    Args:
        yield_proj: Projections DataFrame with yield_tech_trend, climate_impact_bu, acres_harvested.
        land_values: NASS land values with fips, land_value_per_acre.
        discount_rate: Real discount rate.
        horizon: Projection horizon in years.
        scenario: Climate scenario label.

    Returns:
        DataFrame with stranded value per county (aggregated across crops).
    """
    yield_proj = yield_proj.copy()
    yield_proj['price'] = yield_proj['crop'].map(COMMODITY_PRICES).fillna(5.0)
    yield_proj['climate_income_impact'] = (
        yield_proj['climate_impact_bu'] * yield_proj['price']
    )
    yield_proj['climate_income_total'] = (
        yield_proj['climate_income_impact'] * yield_proj['acres_harvested']
    )
    min_year = yield_proj['year'].min()
    yield_proj['years_ahead'] = yield_proj['year'] - min_year + 1
    yield_proj = yield_proj[yield_proj['years_ahead'] <= horizon]
    yield_proj['discount_factor'] = 1.0 / (1 + discount_rate) ** yield_proj['years_ahead']
    yield_proj['pv_climate_impact'] = (
        yield_proj['climate_income_total'] * yield_proj['discount_factor']
    )
    county_pv = (
        yield_proj.groupby('fips')
        .agg(
            pv_climate_total=('pv_climate_impact', 'sum'),
            total_acres=('acres_harvested', 'mean'),
            mean_climate_impact_bu=('climate_impact_bu', 'mean'),
        )
        .reset_index()
    )
    county_pv['stranded_value_total'] = -county_pv['pv_climate_total']
    county_pv['stranded_value_per_acre'] = (
        county_pv['stranded_value_total'] / county_pv['total_acres'].replace(0, np.nan)
    )
    if not land_values.empty:
        land_avg = (
            land_values.groupby('fips')['land_value_per_acre']
            .mean()
            .reset_index()
        )
        county_pv = county_pv.merge(land_avg, on='fips', how='left')
        county_pv['stranded_fraction'] = (
            county_pv['stranded_value_per_acre'] /
            county_pv['land_value_per_acre'].replace(0, np.nan)
        )
    else:
        county_pv['land_value_per_acre'] = np.nan
        county_pv['stranded_fraction'] = np.nan

    county_pv['scenario'] = scenario
    county_pv['discount_rate'] = discount_rate
    county_pv['horizon'] = horizon
    return county_pv


def compute_edd_above_threshold(
    tmax_july_C: np.ndarray,
    tmax_growing_C: np.ndarray,
    threshold_C: float = SR_THRESHOLD_MODERATE,
) -> np.ndarray:
    """Compute extreme degree-days above threshold (Schlenker & Roberts 2009).

    Args:
        tmax_july_C: Array of mean July Tmax in degrees Celsius.
        tmax_growing_C: Array of mean growing-season Tmax (May-Sep) in °C.
        threshold_C: Damage threshold in °C.

    Returns:
        Array of annual EDD values (degree-days above threshold, growing season).
    """
    edd_july = np.maximum(0.0, tmax_july_C - threshold_C) * SR_JULY_DAYS
    edd_shoulder = np.maximum(0.0, tmax_growing_C - threshold_C) * SR_SHOULDER_DAYS
    return edd_july + edd_shoulder


def compute_stranded_with_damage_function(
    yield_proj: pd.DataFrame,
    climate_proj: pd.DataFrame,
    land_values: pd.DataFrame,
    discount_rate: float = 0.03,
    horizon: int = 35,
    scenario: str = 'SSP370',
    indirect_multiplier: float = 1.30,
) -> pd.DataFrame:
    """Compute stranded assets using ML + Schlenker-Roberts (2009) EDD damage function.

    Adds an additive EDD-based yield penalty on top of the ML model estimate.

    Args:
        yield_proj: Yield projections DataFrame.
        climate_proj: County climate projections with tmax_july_projected (°F) and deltas.
        land_values: NASS land values for stranded fraction computation.
        discount_rate: Real discount rate.
        horizon: Projection horizon in years.
        scenario: Climate scenario label.
        indirect_multiplier: Multiplier on combined ML+SR impact (captures indirect losses).

    Returns:
        DataFrame with stranded value per county, decomposed into ML and SR components.
    """
    yield_proj = yield_proj.copy()
    climate_proj = climate_proj.copy()

    # Convert projected temperatures from °F to °C
    climate_proj['tmax_july_C'] = (climate_proj['tmax_july_projected'] - 32) * 5.0 / 9.0
    climate_proj['tmax_growing_C'] = (climate_proj['tmax_growing_projected'] - 32) * 5.0 / 9.0

    # Historical baseline (remove the warming delta)
    climate_proj['tmax_july_baseline_C'] = (
        (climate_proj['tmax_july_projected'] - climate_proj['delta_tmax_july']) - 32
    ) * 5.0 / 9.0
    climate_proj['tmax_growing_baseline_C'] = (
        (climate_proj['tmax_growing_projected'] - climate_proj['delta_tmax_growing']) - 32
    ) * 5.0 / 9.0

    climate_proj['edd_projected'] = compute_edd_above_threshold(
        climate_proj['tmax_july_C'].values,
        climate_proj['tmax_growing_C'].values,
    )
    climate_proj['edd_baseline'] = compute_edd_above_threshold(
        climate_proj['tmax_july_baseline_C'].values,
        climate_proj['tmax_growing_baseline_C'].values,
    )
    climate_proj['delta_edd'] = (
        climate_proj['edd_projected'] - climate_proj['edd_baseline']
    ).clip(lower=0)

    yield_proj['price'] = yield_proj['crop'].map(COMMODITY_PRICES).fillna(5.0)
    yield_proj['sr_coef'] = yield_proj['crop'].map(SR_COEFFICIENTS).fillna(SR_COEFFICIENTS['corn'])

    clim_key = climate_proj[['fips', 'year', 'tmax_july_C', 'tmax_growing_C',
                              'edd_projected', 'delta_edd']]
    yield_proj = yield_proj.merge(clim_key, on=['fips', 'year'], how='left')
    yield_proj['delta_edd'] = yield_proj['delta_edd'].fillna(0.0)

    yield_proj['sr_yield_penalty'] = (
        yield_proj['delta_edd'] * yield_proj['sr_coef']
    )
    yield_proj['climate_impact_combined'] = (
        yield_proj['climate_impact_bu'] + yield_proj['sr_yield_penalty']
    )

    yield_proj['income_ml'] = (
        yield_proj['climate_impact_bu'] * yield_proj['price'] * yield_proj['acres_harvested']
    )
    yield_proj['income_sr_add'] = (
        yield_proj['sr_yield_penalty'] * yield_proj['price'] * yield_proj['acres_harvested']
    )
    yield_proj['income_combined'] = (
        yield_proj['climate_impact_combined'] * yield_proj['price']
        * yield_proj['acres_harvested'] * indirect_multiplier
    )

    min_year = yield_proj['year'].min()
    yield_proj['years_ahead'] = yield_proj['year'] - min_year + 1
    yield_proj = yield_proj[yield_proj['years_ahead'] <= horizon]
    yield_proj['discount_factor'] = 1.0 / (1 + discount_rate) ** yield_proj['years_ahead']

    yield_proj['pv_ml'] = yield_proj['income_ml'] * yield_proj['discount_factor']
    yield_proj['pv_sr_add'] = yield_proj['income_sr_add'] * yield_proj['discount_factor']
    yield_proj['pv_combined'] = yield_proj['income_combined'] * yield_proj['discount_factor']

    county_pv = (
        yield_proj.groupby('fips')
        .agg(
            pv_ml_total=('pv_ml', 'sum'),
            pv_sr_additive=('pv_sr_add', 'sum'),
            pv_combined_total=('pv_combined', 'sum'),
            total_acres=('acres_harvested', 'mean'),
            mean_delta_edd=('delta_edd', 'mean'),
            mean_tmax_july_C=('tmax_july_C', 'mean'),
            mean_sr_yield_penalty=('sr_yield_penalty', 'mean'),
        )
        .reset_index()
    )

    county_pv['stranded_value_total'] = -county_pv['pv_combined_total']
    county_pv['stranded_ml_only'] = -county_pv['pv_ml_total']
    county_pv['stranded_sr_additive'] = -county_pv['pv_sr_additive']
    county_pv['stranded_value_per_acre'] = (
        county_pv['stranded_value_total'] / county_pv['total_acres'].replace(0, np.nan)
    )

    if not land_values.empty:
        land_avg = land_values.groupby('fips')['land_value_per_acre'].mean().reset_index()
        county_pv = county_pv.merge(land_avg, on='fips', how='left')
        county_pv['stranded_fraction'] = (
            county_pv['stranded_value_per_acre'] /
            county_pv['land_value_per_acre'].replace(0, np.nan)
        )
    else:
        county_pv['land_value_per_acre'] = np.nan
        county_pv['stranded_fraction'] = np.nan

    county_pv['scenario'] = scenario
    county_pv['discount_rate'] = discount_rate
    county_pv['horizon'] = horizon
    county_pv['damage_method'] = 'SR_EDD_additive'
    county_pv['indirect_multiplier'] = indirect_multiplier

    return county_pv


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Execute stranded asset computation for SSP3-7.0 and compare with SSP245."""
    logger.info("=" * 60)
    logger.info("STRANDED ASSETS — SSP3-7.0")
    logger.info("=" * 60)

    # Load SSP370 yield projections
    proj_path = PROJECTIONS_DIR / f'yield_projections_{SCENARIO}.parquet'
    yield_proj = pd.read_parquet(proj_path)
    logger.info(f"Loaded SSP370 projections: {len(yield_proj)} rows, "
                f"{yield_proj['fips'].nunique()} counties")

    # Load SSP370 climate projections
    clim_path = PROJECTIONS_DIR / 'county_climate_projections_ssp370.parquet'
    climate_proj = pd.read_parquet(
        clim_path,
        columns=['fips', 'year', 'tmax_july_projected', 'delta_tmax_july',
                 'tmax_growing_projected', 'delta_tmax_growing']
    )
    logger.info(f"Loaded SSP370 climate: {len(climate_proj)} rows")

    # Land values
    land_path = DATA_RAW / 'nass' / 'nass_land_values.parquet'
    land_values = pd.read_parquet(land_path) if land_path.exists() else pd.DataFrame()

    r = CONFIG['stranded_assets']['discount_rate']       # 0.04
    h = CONFIG['stranded_assets']['projection_horizon']  # 30

    # ── Method 1: Conservative — ML only, r=4%, h=30yr ──────────────────────
    national = compute_stranded_vectorized(yield_proj, land_values, r, h, SCENARIO)
    pos_cons = national[national['stranded_value_total'] > 0]
    neg_cons = national[national['stranded_value_total'] <= 0]
    total_cons_B   = pos_cons['stranded_value_total'].sum() / 1e9
    total_gained_B = abs(neg_cons['stranded_value_total'].sum()) / 1e9

    logger.info(f"\nMethod 1 — Conservative (ML only), {SCENARIO}, r={r}, h={h}:")
    logger.info(f"  Counties stranded: {len(pos_cons)}")
    logger.info(f"  Total stranded:    ${total_cons_B:.1f}B")
    logger.info(f"  Total gained:      ${total_gained_B:.1f}B")
    logger.info(f"  Net:               ${(total_cons_B - total_gained_B):.1f}B")

    national.to_parquet(OUTPUT_DIR / f'stranded_national_{SCENARIO}.parquet', index=False)

    # ── Method 2: Central — ML + SR + indirect, r=3%, h=35yr ────────────────
    INDIRECT_MULTIPLIER   = 1.30
    CENTRAL_DISCOUNT_RATE = 0.03
    CENTRAL_HORIZON       = 35

    logger.info(
        f"\nMethod 2 — Central (ML + SR + indirect {INDIRECT_MULTIPLIER}x), "
        f"{SCENARIO}, r={CENTRAL_DISCOUNT_RATE}, h={CENTRAL_HORIZON}:"
    )
    national_sr = compute_stranded_with_damage_function(
        yield_proj, climate_proj, land_values,
        discount_rate=CENTRAL_DISCOUNT_RATE,
        horizon=CENTRAL_HORIZON,
        scenario=SCENARIO,
        indirect_multiplier=INDIRECT_MULTIPLIER,
    )

    pos_sr           = national_sr[national_sr['stranded_value_total'] > 0]
    neg_sr           = national_sr[national_sr['stranded_value_total'] <= 0]
    total_sr_B       = pos_sr['stranded_value_total'].sum() / 1e9
    total_gained_sr_B= abs(neg_sr['stranded_value_total'].sum()) / 1e9
    sr_additive_B    = national_sr['stranded_sr_additive'].clip(lower=0).sum() / 1e9
    mean_delta_edd   = national_sr['mean_delta_edd'].mean()
    mean_tmax        = national_sr['mean_tmax_july_C'].mean()

    logger.info(f"  Mean July Tmax (projected, °C):  {mean_tmax:.2f}")
    logger.info(f"  Mean incremental EDD above 29°C: {mean_delta_edd:.1f} degree-days")
