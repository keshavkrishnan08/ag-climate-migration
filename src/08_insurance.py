"""Phase 5C: Crop insurance mispricing quantification.

Federal crop insurance premiums are based on Actual Production History (APH) —
a county's average yield over the prior 10 years. This backward-looking calculation
was designed for a stationary climate. In a non-stationary climate, it systematically:
    - Underestimates future risk in WARMING counties (premiums too cheap)
    - Overestimates risk in BENEFITING counties (premiums too expensive)

The result: northern counties subsidize southern counties through a cross-subsidy
that is invisible in current policy discussions.

Actuarial logic (PRD Section 7, Computation C):
    Current premium: based on 10-year APH (backward-looking)
    Fair premium: current_premium × EI_ratio
    EI_ratio = E[indemnity | future yield dist] / E[indemnity | APH yield dist]
    Mispricing = current_premium × (EI_ratio - 1)
    Positive = county is UNDERPRICED (too cheap, taxpayer subsidy too large)
    Negative = county is OVERPRICED (too expensive, farmer overpaying)
    Cross-subsidy = min(total_underpriced, total_overpriced) = risk pool transfer

Method: expected indemnity computed via analytical put formula E[max(K-X,0)]
with K = APH × 0.75 × price and X ~ N(mean, sigma²).
Yield CV sourced from 15 years of NASS historical data (2008-2023) to capture
true interannual variability, not climate-model ensemble spread.

Target finding: $3-8B/yr total structural mispricing, $1-3B/yr cross-subsidy.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
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

# RMA crop_name → model crop name mapping (after strip+upper)
RMA_CROP_MAP = {
    'CORN': 'corn',
    'SOYBEANS': 'soybeans',
    'WHEAT': 'wheat_winter',
    'COTTON': 'cotton',
    'GRAIN SORGHUM': 'sorghum',
    'BARLEY': 'barley',
    'OATS': 'oats',
}

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


def compute_aph_premium(
    historical_yields: np.ndarray,
    price: float,
    coverage: float = 0.75,
    loading_factor: float = 1.15
) -> float:
    """Compute current RMA-style premium from Actual Production History.

    APH uses the most recent 10 years of yields to set the guarantee
    and estimate the loss distribution.

    Args:
        historical_yields: Array of last 10 years of yields (bu/acre).
        price: Projected price for the crop year ($/bu).
        coverage: Coverage level (0.50 to 0.85, standard is 0.75).
        loading_factor: Administrative/operating expense load.

    Returns:
        Premium per acre in dollars.
    """
    if len(historical_yields) == 0:
        return 0.0

    aph_yield = np.mean(historical_yields)
    guarantee = aph_yield * coverage * price

    # Estimate loss distribution from historical variance
    std = np.std(historical_yields)
    cv = std / aph_yield if aph_yield > 0 else 0.15

    # Expected indemnity: E[max(guarantee - actual_yield × price, 0)]
    # Using lognormal approximation
    if cv > 0 and aph_yield > 0:
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(aph_yield) - sigma**2 / 2

        # Monte Carlo estimate of expected indemnity
        np.random.seed(CONFIG['yield_model']['random_seed'])
        simulated_yields = np.random.lognormal(mu, sigma, 10000)
        simulated_revenue = simulated_yields * price
        indemnities = np.maximum(guarantee - simulated_revenue, 0)
        expected_indemnity = np.mean(indemnities)
    else:
        expected_indemnity = 0

    premium = expected_indemnity * loading_factor
    return float(premium)


def compute_fair_premium(
    projected_yield_dist: np.ndarray,
    price: float,
    coverage: float = 0.75,
    loading_factor: float = 1.15
) -> float:
    """Compute actuarially fair premium from forward-looking yield distribution.

    Uses projected yield distribution under climate scenario rather than
    backward-looking APH.

    Args:
        projected_yield_dist: Array of simulated future yields (n=1000+).
        price: Projected price ($/bu).
        coverage: Coverage level.
        loading_factor: Administrative load.

    Returns:
        Fair premium per acre in dollars.
    """
    if len(projected_yield_dist) == 0:
        return 0.0

    expected_yield = np.mean(projected_yield_dist)
    guarantee = expected_yield * coverage * price

    # Expected indemnity under projected distribution
    simulated_revenue = projected_yield_dist * price
    indemnities = np.maximum(guarantee - simulated_revenue, 0)
    expected_indemnity = np.mean(indemnities)

    fair_premium = expected_indemnity * loading_factor
    return float(fair_premium)


def simulate_yield_distribution(
    county_fips: str,
    crop: str,
    yield_model,
    climate_scenario: pd.DataFrame,
    n: int = 1000
) -> np.ndarray:
    """Simulate yield distribution under projected climate for a county.

    Args:
        county_fips: 5-digit FIPS.
        crop: Crop type.
        yield_model: Trained yield model.
        climate_scenario: Projected climate trajectory.
        n: Number of simulations.

    Returns:
        Array of n simulated yields.
    """
    np.random.seed(CONFIG['yield_model']['random_seed'])

    # Get projected mean yield and uncertainty
    county_proj = climate_scenario[
        (climate_scenario['fips'] == county_fips) &
        (climate_scenario.get('crop', '') == crop)
    ] if 'crop' in climate_scenario.columns else pd.DataFrame()

    if county_proj.empty:
        return np.array([])

    mean_yield = county_proj['yield_projected'].mean()
    yield_std = county_proj['yield_projected'].std()

    if yield_std == 0:
        yield_std = mean_yield * 0.15  # Default CV

    # Simulate from lognormal (yields can't be negative)
    if mean_yield > 0:
        sigma = np.sqrt(np.log(1 + (yield_std / mean_yield)**2))
        mu = np.log(mean_yield) - sigma**2 / 2
        simulated = np.random.lognormal(mu, sigma, n)
    else:
        simulated = np.zeros(n)

    return simulated


def compute_insurance_mispricing(
    county_fips: str,
    crop: str,
    rma_data: pd.DataFrame,
    yield_projections: pd.DataFrame,
    scenario: str = 'SSP245'
) -> dict:
    """Compare current RMA premium to actuarially fair premium.

    RMA data has fips as a proper column and crop_name already normalized
    (stripped and lowercased to match model crop names) before being passed in.

    Args:
        county_fips: 5-digit FIPS code.
        crop: Crop type string (normalized, e.g. 'corn').
        rma_data: USDA RMA Summary of Business with normalized crop_name.
        yield_projections: Projected yields under scenario.
        scenario: Climate scenario name.

    Returns:
        Dict with mispricing_per_acre, direction, annual_cross_subsidy.
    """
    # Step 1: Current RMA premium from actual data
    # rma_data has been pre-filtered with normalized crop names in 'crop' column
    if not rma_data.empty and 'fips' in rma_data.columns:
        county_rma = rma_data[
            (rma_data['fips'] == county_fips) &
            (rma_data['crop'] == crop)
        ]
    else:
        county_rma = pd.DataFrame()

    if not county_rma.empty and 'premium_per_acre' in county_rma.columns:
        current_premium = county_rma['premium_per_acre'].dropna().mean()
        if np.isnan(current_premium):
            current_premium = 0.0
    else:
        current_premium = 0.0

    # Step 2: Compute fair premium from projected yield distribution
    if not yield_projections.empty:
        county_proj = yield_projections[
            (yield_projections['fips'] == county_fips) &
            (yield_projections['crop'] == crop)
        ]
    else:
        county_proj = pd.DataFrame()

    price = COMMODITY_PRICES.get(crop, 5.0)

    if not county_proj.empty and 'yield_projected' in county_proj.columns:
        projected_yields = county_proj['yield_projected'].dropna().values
        if len(projected_yields) > 0:
            mean_y = np.mean(projected_yields)
            std_y = np.std(projected_yields) if len(projected_yields) > 1 else mean_y * 0.15
            if mean_y > 0 and std_y > 0:
                sigma = np.sqrt(np.log(1 + (std_y / mean_y) ** 2))
                mu = np.log(mean_y) - sigma ** 2 / 2
                np.random.seed(CONFIG['yield_model']['random_seed'])
                sim_yields = np.random.lognormal(mu, sigma, 1000)
            else:
                sim_yields = np.full(1000, max(mean_y, 0.0))
            fair_premium = compute_fair_premium(sim_yields, price)
        else:
            fair_premium = current_premium
    else:
        fair_premium = current_premium

    # Step 3: Compute mispricing
    mispricing = fair_premium - current_premium

    # Step 4: Annual aggregate dollar flow — use 'acres' column (RMA has no 'liability_acres')
    if not county_rma.empty and 'acres' in county_rma.columns:
        insured_acres = county_rma['acres'].dropna().mean()
        if np.isnan(insured_acres):
            insured_acres = 0.0
    else:
        insured_acres = 0.0

    annual_flow = mispricing * insured_acres

    # Direction
    if mispricing > 0:
        direction = 'underpriced'   # Too cheap → taxpayer subsidy too large
    elif mispricing < 0:
        direction = 'overpriced'    # Too expensive → farmer overpaying
    else:
        direction = 'fair'

    return {
        'fips': county_fips,
        'crop': crop,
        'scenario': scenario,
        'current_premium_per_acre': float(current_premium),
        'fair_premium_per_acre': float(fair_premium),
        'mispricing_per_acre': float(mispricing),
        'direction': direction,
        'insured_acres': float(insured_acres),
        'annual_cross_subsidy': float(annual_flow),
    }


def normalize_rma_crops(rma_data: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and map RMA crop_name to model crop names.

    RMA crop_name values have trailing whitespace and uppercase text.
    This normalizes them and adds a 'crop' column for joining with projections.

    Args:
        rma_data: Raw RMA DataFrame with crop_name column.

    Returns:
        RMA DataFrame with added 'crop' column (normalized), filtered to known crops.
    """
    rma = rma_data.copy()
    rma['crop_name_clean'] = rma['crop_name'].str.strip().str.upper()
    rma['crop'] = rma['crop_name_clean'].map(RMA_CROP_MAP)
    # Keep only rows that map to a known model crop
    rma = rma[rma['crop'].notna()].copy()
    logger.info(f"RMA rows after crop normalization: {len(rma)} ({rma['crop'].nunique()} crops)")
    return rma


def _compute_yield_cv_from_nass(nass_path: Path) -> pd.DataFrame:
    """Compute county-crop yield coefficient of variation from NASS historical data.

    Uses the last 15 years of observed yields (2008-2023) to estimate interannual
    yield variability — the true actuarial risk parameter. The P10/P90 spread in
    the climate projections represents model-ensemble uncertainty, not interannual
    variability, so we anchor to observed historical data instead.

    Args:
        nass_path: Path to NASS county yields parquet.

    Returns:
        DataFrame with columns [fips, crop, yield_cv] where yield_cv is clipped
        to [0.05, 0.50] to prevent degenerate distributions.
    """
    try:
        nass = pd.read_parquet(
            nass_path,
            columns=['fips', 'year', 'crop', 'yield_bu_acre']
        )
    except Exception:
        logger.warning("Could not load NASS yields for CV estimation — using crop defaults")
        return pd.DataFrame(columns=['fips', 'crop', 'yield_cv'])

    nass_recent = nass[
        nass['year'].between(2008, 2023) & (nass['yield_bu_acre'] > 0)
    ].copy()

    hist = (
        nass_recent
        .groupby(['fips', 'crop'])
        .agg(hist_mean=('yield_bu_acre', 'mean'),
             hist_std=('yield_bu_acre', 'std'),
             n_obs=('yield_bu_acre', 'count'))
        .reset_index()
    )
    # Need at least 5 observations for a reliable CV estimate
    hist = hist[hist['n_obs'] >= 5].copy()
    hist['yield_cv'] = (hist['hist_std'] / hist['hist_mean']).clip(0.05, 0.50)
    hist['yield_cv'] = hist['yield_cv'].fillna(0.20)

    logger.info(
        f"Historical yield CV computed for {len(hist)} county-crop pairs "
        f"(median={hist['yield_cv'].median():.3f})"
    )
    return hist[['fips', 'crop', 'yield_cv']]


def _expected_indemnity(K: float, mu: float, sigma: float) -> float:
    """Expected indemnity for a revenue guarantee contract (analytical put formula).

    Computes E[max(K - X, 0)] where X ~ N(mu, sigma^2), representing expected
    indemnity payment per acre when the guarantee is K and revenue is normally
    distributed.

    Args:
        K: Guarantee level (= APH_yield × coverage × price), $/acre.
        mu: Expected revenue under assumed yield distribution, $/acre.
        sigma: Revenue standard deviation (yield_std × price), $/acre.

    Returns:
        Expected indemnity per acre in dollars (always >= 0).
    """
    sigma = max(sigma, 1.0)
    z = (K - mu) / sigma
    ei = (K - mu) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
    return float(max(ei, 0.0))


def compute_national_mispricing(
    rma_data: pd.DataFrame,
    yield_projections: pd.DataFrame,
    scenario: str = 'SSP245'
) -> pd.DataFrame:
    """Compute insurance mispricing for all county-crop pairs nationally.

    Uses an actuarially grounded expected-indemnity approach:

    1. APH yield = yield_baseline (backward-looking 10-year mean embedded in projections).
    2. Future yield = mean projected yield 2040-2050 under the given climate scenario.
    3. Yield variability = county-crop CV from 15 years of NASS historical data (2008-2023),
       which captures true interannual risk rather than model-ensemble spread.
    4. Expected indemnity ratio = EI(future) / EI(APH), computed via the analytical put
       formula E[max(K - X, 0)] with K = APH × 0.75 × price and X ~ N(mean, sigma).
    5. Mispricing per acre = current_premium_per_acre × (EI_ratio - 1).
       Positive = underpriced (county risk underweighted); negative = overpriced.
    6. Cross-subsidy = total flow from overpriced counties to underpriced counties
       through the shared federal risk pool = min(total_underpriced, total_overpriced).

    Data quality filters applied:
    - Future yield clipped at zero (projection artifacts).
    - Minimum APH thresholds by crop to exclude fringe-production counties.
    - EI ratio capped at MAX_EI_RATIO (default 5×) to prevent tail outliers.

    Args:
        rma_data: RMA Summary of Business (raw, with crop_name column).
        yield_projections: Projected yields DataFrame (columns: fips, year, crop,
            yield_projected, yield_baseline, acres_harvested).
        scenario: Climate scenario label.

    Returns:
        DataFrame with mispricing analysis per county-crop pair, including columns:
        fips, crop, scenario, aph_yield, future_yield, yield_delta_pct, yield_cv,
        direction, ei_ratio, mispricing_per_acre, insured_acres, annual_cross_subsidy.
    """
    FUTURE_START = 2040
    FUTURE_END = 2050
    COVERAGE = 0.75          # Standard RP-HPE coverage level
    MAX_EI_RATIO = 5.0       # Cap on EI ratio to prevent tail distortion
    # Minimum APH by crop to exclude marginal/fringe production counties
    CROP_MIN_APH = {
        'corn': 50.0, 'soybeans': 10.0, 'wheat_winter': 5.0,
        'cotton': 100.0, 'sorghum': 10.0, 'barley': 10.0, 'oats': 5.0,
    }

    logger.info(f"Computing national insurance mispricing under {scenario} (EI-ratio method)...")

    # ------------------------------------------------------------------ #
    # 1. RMA: normalize crop names, aggregate to county-crop (last 10 yr) #
    # ------------------------------------------------------------------ #
    if not rma_data.empty:
        rma_norm = normalize_rma_crops(rma_data)
        rma_recent = rma_norm[rma_norm['year'] >= rma_norm['year'].max() - 10]

        # IMPORTANT: each county-crop-year has many rows (one per plan/coverage level).
        # Sum acres and total_premium across all plans first, THEN average across years.
        # Using mean(acres) directly would severely under-count insured acres.
        rma_by_year = (
            rma_recent
            .groupby(['fips', 'crop', 'year'], as_index=False)
            .agg(
                acres_yr=('acres', 'sum'),
                premium_total_yr=('total_premium', 'sum'),
                indemnity_total_yr=('indemnity', 'sum'),
            )
        )
        rma_agg = (
            rma_by_year
            .groupby(['fips', 'crop'], as_index=False)
            .agg(
                insured_acres=('acres_yr', 'mean'),
                total_premium=('premium_total_yr', 'mean'),
                avg_indemnity=('indemnity_total_yr', 'mean'),
            )
        )
        # Derive premium_per_acre from totals to preserve weighting across plan types
        rma_agg['premium_per_acre'] = (
            rma_agg['total_premium'] / rma_agg['insured_acres'].replace(0, np.nan)
        )
        # Drop rows with zero or missing insured acres
        rma_agg = rma_agg[rma_agg['insured_acres'] > 0].copy()
        logger.info(
            f"RMA county-crop pairs (last 10yr, >0 acres, summed across plans): {len(rma_agg)}"
        )
    else:
        rma_agg = pd.DataFrame(
            columns=['fips', 'crop', 'premium_per_acre', 'insured_acres', 'avg_indemnity']
        )
        logger.warning("RMA data empty — mispricing magnitudes will be zero")

    # ------------------------------------------------------------------ #
    # 2. Historical yield CV from NASS (interannual variability)           #
    # ------------------------------------------------------------------ #
    nass_path = DATA_RAW / 'nass' / 'nass_county_yields.parquet'
    yield_cv_df = _compute_yield_cv_from_nass(nass_path)
    # Crop-level median CV as fallback for counties missing NASS data
    crop_median_cv: dict = {}
    if not yield_cv_df.empty:
        crop_median_cv = yield_cv_df.groupby('crop')['yield_cv'].median().to_dict()

    # ------------------------------------------------------------------ #
    # 3. Projections: APH yield from baseline; future from 2040-2050 mean #
    # ------------------------------------------------------------------ #
    if yield_projections.empty:
        logger.error("No yield projections — cannot compute mispricing")
        return pd.DataFrame()

    # APH: yield_baseline is the backward-looking mean the model was anchored to.
    aph = (
        yield_projections
        .groupby(['fips', 'crop'], as_index=False)
        .agg(aph_yield=('yield_baseline', 'mean'))
    )

    # Future: mean projected yield over 2040-2050
    future = (
        yield_projections[yield_projections['year'].between(FUTURE_START, FUTURE_END)]
        .groupby(['fips', 'crop'], as_index=False)
        .agg(future_yield=('yield_projected', 'mean'))
    )

    proj = aph.merge(future, on=['fips', 'crop'], how='inner')
    logger.info(f"County-crop pairs with APH + future projections: {len(proj)}")

