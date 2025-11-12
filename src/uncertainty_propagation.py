"""Monte Carlo uncertainty propagation for stranded asset DCF estimate.

Propagates yield model uncertainty (R²=0.21) through the DCF stranded asset
computation using 1,000 Monte Carlo draws from the residual distribution.

Outputs:
    results/stranded_assets/uncertainty_propagation.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
RESULTS_DIR = PROJECT_ROOT / 'results'
PROJECTIONS_DIR = PROJECT_ROOT / 'data' / 'projections'

# Yield model test R² from v2 model (results/yield_model_v2_metrics.json)
R2_YIELD = 0.2059691810308517

# Commodity prices (2023 USD) — must match 06_stranded.py
COMMODITY_PRICES = {
    'corn': 5.50, 'soybeans': 12.80, 'wheat_winter': 7.20,
    'wheat_spring': 8.10, 'cotton': 0.78, 'sorghum': 5.30,
    'barley': 6.10, 'oats': 3.80,
}

# DCF parameters: baseline (4%, 30yr) — conservative scenario
DISCOUNT_RATE = 0.04
HORIZON = 30
N_ITER = 1000
SEED = 42


def compute_total_stranded(yield_proj: pd.DataFrame) -> float:
    """Compute total national stranded value ($ billions) from yield projections.

    Args:
        yield_proj: DataFrame with climate_impact_bu, acres_harvested, year, crop.

    Returns:
        Total stranded value in billions USD (positive = loss).
    """
    df = yield_proj.copy()
    df['price'] = df['crop'].map(COMMODITY_PRICES).fillna(5.0)
    df['climate_income_total'] = df['climate_impact_bu'] * df['price'] * df['acres_harvested']

    min_year = df['year'].min()
    df['years_ahead'] = df['year'] - min_year + 1
    df = df[df['years_ahead'] <= HORIZON]
    df['discount_factor'] = 1.0 / (1 + DISCOUNT_RATE) ** df['years_ahead']
    df['pv_climate_impact'] = df['climate_income_total'] * df['discount_factor']

    # County-level stranded = -PV(climate impact)
    county_pv = df.groupby('fips')['pv_climate_impact'].sum().reset_index()
    county_pv['stranded'] = -county_pv['pv_climate_impact']
    total_b = county_pv[county_pv['stranded'] > 0]['stranded'].sum() / 1e9
    return total_b


def run_monte_carlo() -> dict:
    """Run 1,000-iteration Monte Carlo propagation of yield model uncertainty.

    For each iteration:
        1. Compute residual_std = sqrt(1 - R²) * std(climate_impact_bu)
           This is the standard deviation of yield prediction errors in bu/acre units.
        2. Add N(0, residual_std) noise to climate_impact_bu (all rows independently).
        3. Recompute stranded asset total with noisy predictions.
        4. Record total stranded.

    Returns:
        Dict with mean, p2.5, p97.5, and all iteration values.
    """
    logger.info("Loading yield projections SSP2-4.5...")
    yp = pd.read_parquet(
        PROJECTIONS_DIR / 'yield_projections_SSP245.parquet',
        columns=['fips', 'year', 'crop', 'climate_impact_bu', 'acres_harvested']
    )
    yp['fips'] = yp['fips'].astype(str).str.zfill(5)

    # Residual std: sqrt(1 - R²) * std(target)
    # climate_impact_bu is already the yield anomaly in bu/acre — that's our target
    impact_std = yp['climate_impact_bu'].std()
    residual_std = np.sqrt(1.0 - R2_YIELD) * impact_std

    logger.info(
        f"climate_impact_bu std={impact_std:.3f} bu/ac, "
        f"R²={R2_YIELD:.4f}, residual_std={residual_std:.3f} bu/ac"
    )

    # Point estimate (no noise)
    point_estimate = compute_total_stranded(yp)
    logger.info(f"Point estimate: ${point_estimate:.2f}B")

    rng = np.random.default_rng(SEED)
    totals = []

    for i in range(N_ITER):
        yp_noisy = yp.copy()
        noise = rng.normal(0.0, residual_std, size=len(yp_noisy))
        yp_noisy['climate_impact_bu'] = yp_noisy['climate_impact_bu'] + noise
        total_b = compute_total_stranded(yp_noisy)
        totals.append(total_b)
        if (i + 1) % 100 == 0:
            logger.info(f"  Iteration {i+1}/{N_ITER} — running mean: ${np.mean(totals):.2f}B")

    totals = np.array(totals)
    result = {
        "method": "Monte Carlo uncertainty propagation",
        "n_iterations": N_ITER,
        "yield_model_r2": R2_YIELD,
        "residual_std_bu_acre": float(residual_std),
        "discount_rate": DISCOUNT_RATE,
        "horizon": HORIZON,
        "scenario": "SSP245",
        "point_estimate_b": float(point_estimate),
        "mean_b": float(np.mean(totals)),
        "median_b": float(np.median(totals)),
        "p2_5_b": float(np.percentile(totals, 2.5)),
        "p97_5_b": float(np.percentile(totals, 97.5)),
        "p5_b": float(np.percentile(totals, 5)),
        "p95_b": float(np.percentile(totals, 95)),
        "std_b": float(np.std(totals)),
        "seed": SEED,
    }
    logger.info(
        f"MC result: mean=${result['mean_b']:.1f}B, "
        f"95% CI=[${result['p2_5_b']:.1f}B, ${result['p97_5_b']:.1f}B]"
    )
    return result


def main():
    """Run propagation and save results."""
    out_dir = RESULTS_DIR / 'stranded_assets'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'uncertainty_propagation.json'

    result = run_monte_carlo()

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved uncertainty propagation to {out_path}")

    print(f"\n=== Uncertainty Propagation Results ===")
    print(f"Point estimate:  ${result['point_estimate_b']:.1f}B")
    print(f"MC mean:         ${result['mean_b']:.1f}B")
    print(f"95% CI:          [${result['p2_5_b']:.1f}B, ${result['p97_5_b']:.1f}B]")
    print(f"R² = {result['yield_model_r2']:.2f}, residual_std = {result['residual_std_bu_acre']:.2f} bu/ac")
    return result


if __name__ == '__main__':
    main()
