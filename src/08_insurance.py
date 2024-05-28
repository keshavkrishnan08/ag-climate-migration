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

