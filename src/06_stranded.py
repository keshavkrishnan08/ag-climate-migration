"""Phase 5A: Stranded agricultural asset valuation.

Computes the present discounted value gap between farmland valued under
a no-climate-change trajectory and farmland valued under projected climate.

    Stranded value = PV(income under tech trend only) - PV(income under tech + climate)
    Positive = county loses value due to climate (stranded asset)
    Negative = county gains value (climate benefit)

Reviewer Fix 3: Sensitivity grid (discount 2-8% x horizon 20-40yr) + cap rate method.

Enhancement: Schlenker-Roberts (2009) non-linear damage function + synthetic SSP5-8.5.
"""

import json
import sys
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

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

COMMODITY_PRICES = {
    'corn': 5.50, 'soybeans': 12.80, 'wheat_winter': 7.20,
    'wheat_spring': 8.10, 'cotton': 0.78, 'sorghum': 5.30,
    'barley': 6.10, 'oats': 3.80,
}

# Schlenker-Roberts (2009) temperature thresholds (°C)
SR_THRESHOLD_MODERATE = 29.0   # yield response accelerates above this
SR_THRESHOLD_SEVERE = 33.0     # severe damage threshold

# SSP5-8.5 scaling factor relative to SSP2-4.5 (IPCC AR6, warming by 2050)
SSP585_SCALE = 1.8


def compute_stranded_vectorized(
    yield_proj: pd.DataFrame,
    land_values: pd.DataFrame,
    discount_rate: float = 0.04,
    horizon: int = 30,
    scenario: str = 'SSP245'
) -> pd.DataFrame:
    """Compute stranded assets vectorized across all counties/crops.

    Stranded = PV(tech-only income stream) - PV(tech+climate income stream)
             = -PV(climate impact income stream)

    The projections file has:
      yield_tech_trend: yield under technology trend only (no climate change)
      yield_projected: yield under technology + climate impact
      climate_impact_bu: yield_projected - yield_tech_trend (the pure climate effect)
      acres_harvested: county-crop acreage

    Args:
        yield_proj: Projections DataFrame with yield_tech_trend, yield_projected,
                    climate_impact_bu, acres_harvested columns.
        land_values: NASS land values with fips, land_value_per_acre.
        discount_rate: Real discount rate.
        horizon: Projection horizon in years.
        scenario: Climate scenario label.

    Returns:
        DataFrame with stranded value per county (aggregated across crops).
    """
    # Map crop prices
    yield_proj = yield_proj.copy()
    yield_proj['price'] = yield_proj['crop'].map(COMMODITY_PRICES).fillna(5.0)

    # Climate-driven income impact per acre per year
    yield_proj['climate_income_impact'] = (
        yield_proj['climate_impact_bu'] * yield_proj['price']
    )

    # Total climate-driven income impact (income impact x acres)
    yield_proj['climate_income_total'] = (
        yield_proj['climate_income_impact'] * yield_proj['acres_harvested']
    )

    # Discount factor for each year
    min_year = yield_proj['year'].min()
    yield_proj['years_ahead'] = yield_proj['year'] - min_year + 1
    yield_proj = yield_proj[yield_proj['years_ahead'] <= horizon]
    yield_proj['discount_factor'] = 1.0 / (1 + discount_rate) ** yield_proj['years_ahead']

    # PV of climate impact per county-crop
    yield_proj['pv_climate_impact'] = (
        yield_proj['climate_income_total'] * yield_proj['discount_factor']
    )

    # Aggregate to county level (sum across crops and years)
    county_pv = (
        yield_proj.groupby('fips')
        .agg(
            pv_climate_total=('pv_climate_impact', 'sum'),
            total_acres=('acres_harvested', 'mean'),  # avg across years
            mean_climate_impact_bu=('climate_impact_bu', 'mean'),
        )
        .reset_index()
    )

    # Stranded = -PV(climate impact)
    # If climate impact is negative (yield decline), stranded is positive
    county_pv['stranded_value_total'] = -county_pv['pv_climate_total']
    county_pv['stranded_value_per_acre'] = (
        county_pv['stranded_value_total'] / county_pv['total_acres'].replace(0, np.nan)
    )

    # Merge with land values for stranded fraction
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



# Schlenker & Roberts (2009, PNAS) Table 1 — OLS coefficients for US field crops.
# Units: yield loss in bushels/acre per extreme degree-day (EDD) above 29°C.
# EDD = sum of daily max temps above threshold, in degree-day units.
# We approximate annual EDD from July Tmax using 31 days * excess degrees.
# For soybeans: -0.0560 bu/ac/EDD (SR Table 1, col 4).
# For cotton, sorghum, other: use corn coefficient (conservative).
SR_COEFFICIENTS = {
    'corn':         -0.0662,   # SR Table 1 col 1, EDD>29C
