"""Phase 5B: Community collapse cascade.

Uses econometric literature's estimated elasticities to propagate yield decline
into community-level outcomes. Not a standalone ML model — it uses ML yield
projections as input to a structured cascade.

Cascade structure (PRD Section 5.3 & 7B):
    Step 1: Yield decline → Farm income decline
            ΔIncome = Σ_crops [ΔYield × Acres × Price × (1 - InsuranceOffset)]
    Step 2: Farm income → Rural outmigration
            ΔPop = elasticity × ΔIncome% (lagged 3 years)
    Step 3: Outmigration → School enrollment decline
            ΔEnrollment = -0.25 × ΔPop (empirical from NCES, contemporaneous)
    Step 4: Population → Hospital viability
            Closure threshold: county pop < 15,000
    Step 5: Farm income + Population → Tax base
            ΔTaxBase = ΔFarmIncome × 0.35 + ΔPop × AvgPerCapitaTax
    Step 6: Tax base → Infrastructure
            ΔRoadCondition = f(ΔTaxBase) (lagged 5 years)
    Step 7: Infrastructure → Further yield loss (FEEDBACK LOOP)
            Feedback_multiplier = 0.08 per σ decline in infrastructure

Tipping point: county crosses when ALL FOUR conditions met simultaneously:
    1. Population below hospital threshold
    2. School enrollment below closure threshold
    3. Infrastructure feedback accelerating yield loss
    4. Outmigration > 2× in-migration

Target finding: 300 counties cross tipping point before 2040 under RCP 4.5.

Reviewer Fix 4: Re-estimate migration elasticity via IV on 2000-2020 data.
Dual calibration (Reviewer Fix — Issue 2):
    Calibration A: Own IV estimate β=-0.003 (p=0.019, F=1184) — PRIMARY
    Calibration B: Feng et al. (2010) β=-0.17 — SENSITIVITY
    Both are reported so reviewers can evaluate the 57x difference in magnitude.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
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

CASCADE = CONFIG['community_cascade']
COMMODITY_PRICES = {
    'corn': 5.50, 'soybeans': 12.80, 'wheat_winter': 7.20,
    'wheat_spring': 8.10, 'cotton': 0.78, 'sorghum': 5.30,
    'barley': 6.10, 'oats': 3.80,
}


def compute_farm_income_change(
    yield_projections: pd.DataFrame,
    yield_baseline: pd.DataFrame,
    acres_data: pd.DataFrame,
    county_fips: str
) -> pd.DataFrame:
    """Step 1: Compute farm income change from yield projections.

    ΔIncome = Σ_crops [ΔYield × Acres × Price × (1 - InsuranceOffset)]

    Args:
        yield_projections: Projected yields under scenario.
        yield_baseline: Baseline yields.
        acres_data: County-crop acreage.
        county_fips: 5-digit FIPS code.

    Returns:
        DataFrame with annual income change for this county.
    """
    county_proj = yield_projections[yield_projections['fips'] == county_fips]
    county_base = yield_baseline[yield_baseline['fips'] == county_fips]

    income_changes = []
    for year in county_proj['year'].unique():
        year_proj = county_proj[county_proj['year'] == year]
        total_delta_income = 0

        for _, row in year_proj.iterrows():
            crop = row['crop']
            yield_proj = row.get('yield_projected', 0)

            base_row = county_base[county_base['crop'] == crop]
            yield_base = base_row['yield_projected'].mean() if not base_row.empty else yield_proj

            acres_row = acres_data[
                (acres_data['fips'] == county_fips) &
                (acres_data['crop'] == crop)
            ]
            acres = acres_row['acres_harvested'].mean() if not acres_row.empty else 0

            price = COMMODITY_PRICES.get(crop, 5.0)
            delta_yield = yield_proj - yield_base
            delta_income = delta_yield * acres * price * (1 - CONFIG['insurance']['subsidy_rate_avg'])
            total_delta_income += delta_income

        income_changes.append({
            'fips': county_fips,
            'year': year,
            'delta_farm_income': total_delta_income,
        })

    return pd.DataFrame(income_changes)


def compute_population_change(
    income_changes: pd.DataFrame,
    baseline_pop: float,
    elasticity: float = None,
    lag_years: int = 3,
    per_capita_income: float = 35000.0,
) -> pd.DataFrame:
    """Step 2: Compute rural population change from farm income change.

    ΔPop = elasticity × ΔIncome% (for working-age adults in rural Corn Belt)
    Effect is lagged 3-5 years.

    Fix D: Use a direct income-share formula.  Feng et al. (2010) estimate
    the elasticity of outmigration with respect to farm income as a share of
    **farm income** (not total county income).  We use:
        delta_pop_pct = elasticity * (delta_farm_income / baseline_farm_income)
    where baseline_farm_income is proxied as the mean positive farm income
    across all years in income_changes (when available) or falls back to
    baseline_pop × per_capita_income × farm_income_share (≈ 10% of total
    income in rural Corn Belt counties).

    Args:
        income_changes: Annual farm income changes (delta from baseline).
        baseline_pop: County baseline population.
        elasticity: Income-migration elasticity (default from config: -0.17).
        lag_years: Lag between income shock and migration.
        per_capita_income: Rural per-capita income; used only as fallback to
            estimate baseline farm income when no positive income rows exist.
            From census median_household_income or $35,000 rural average.

