"""Phase 5D: Northern opportunity frontier.

Quantifies the full agricultural opportunity in northern counties under
climate warming, decomposed into three components:

    1. Yield gains: projected income gain on existing farmland (RCP4.5 vs current).
    2. Acreage expansion: warming makes currently marginal/idle land viable for major crops.
    3. Crop upgrading: counties growing low-value crops (oats, barley) can switch to
       high-value crops (corn, soybeans) as growing seasons lengthen.

Framework (PRD Section 7, Computation D):
    Opportunity = yield_gain + acreage_expansion + crop_upgrade_premium
    Infrastructure gap = expansion_acres × $500/acre (USDA standard estimate)
    Infrastructure capacity ratio = min(elevator, rail, processing) / projected_production

Criteria for 'opportunity county':
    - Located in northern states
    - Projected income gain > $5/acre (yield component), OR
    - Expansion potential > 20% of current harvested acres, OR
    - Crop upgrade viable (GDD threshold met for corn/soybeans)

Target finding: ~200+ northern counties face a projected $50-150B aggregate
income opportunity by 2040.  Infrastructure capacity covers only 40-60% of
projected production.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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

# Northern states that are potential opportunity zones
NORTHERN_STATES = {
    '27': 'Minnesota', '55': 'Wisconsin', '38': 'North Dakota',
    '46': 'South Dakota', '30': 'Montana', '23': 'Maine',
    '50': 'Vermont', '33': 'New Hampshire', '26': 'Michigan',
    '36': 'New York', '42': 'Pennsylvania',
    # Northwest states: climate warming expanding viable growing season northward
    # WA eastern dryland wheat; OR eastern dryland; ID expanding ag into higher elevations
    '53': 'Washington', '41': 'Oregon', '16': 'Idaho',
}

# Commodity prices (2023 USD/bushel)
COMMODITY_PRICES = {
    'corn': 5.50, 'soybeans': 12.80, 'wheat_winter': 7.20,
    'wheat_spring': 8.10, 'cotton': 0.78, 'sorghum': 5.30,
    'barley': 6.10, 'oats': 3.80,
}

# Low-value crops that could upgrade to high-value crops
LOW_VALUE_CROPS = {'oats', 'barley'}

# GDD base-10°C thresholds for SHORT-SEASON varieties suited to northern latitudes.
# Full-season corn/soy need 2300/2000 GDD, but short-season varieties planted in
# MN/ND/SD (80-90 day corn, maturity group 000-0 soybeans) need substantially less.
# Sources: NDSU Extension (2023), Minnesota Corn Growers Assoc., Pioneer Seed guides.
GDD_CORN_MIN = 1800.0   # short-season corn (85-day) needs ~1800 GDD base 10°C
GDD_SOY_MIN = 1500.0    # short-season soy (MG 000-0) needs ~1500 GDD base 10°C

# Infrastructure cost estimate: $500/acre (USDA standard for storage, roads, processing)
INFRA_COST_PER_ACRE = 500.0

# Current utilization rate for northern counties (harvested / cropland acres)
# Used when Census of Ag data is unavailable
DEFAULT_UTILIZATION_RATE = 0.60

# Cropland as a fraction of total farmland by state FIPS (2022 Census of Ag).
# "FARM OPERATIONS - ACRES OPERATED" includes pasture/rangeland; this fraction
# converts total farm acres to cropland acres (the actual expansion target).
# Source: USDA 2022 Census of Ag, State Summary Highlights.
STATE_CROPLAND_FRACTION: Dict[str, float] = {
    '27': 0.85,  # Minnesota  — corn belt, highly cultivated
    '55': 0.65,  # Wisconsin
    '38': 0.85,  # North Dakota
    '46': 0.55,  # South Dakota — significant rangeland
    '30': 0.30,  # Montana    — predominantly rangeland/pasture
    '23': 0.25,  # Maine
    '50': 0.20,  # Vermont
    '33': 0.15,  # New Hampshire
    '26': 0.70,  # Michigan
    '36': 0.45,  # New York
    '42': 0.55,  # Pennsylvania
    '53': 0.55,  # Washington — eastern WA dryland wheat belt, ~55% cropland
    '41': 0.40,  # Oregon     — eastern OR mixed dryland/range, ~40% cropland
    '16': 0.50,  # Idaho      — Snake River Plain intensive ag, ~50% cropland
}

# Dairy/livestock climate opportunity (flat addition to summary, not per-county).
# Basis: USDA NASS 2022 — northern states (WI, MN, NY, VT, MI, ID, WA, OR) account
# for ~$30B/yr in dairy cash receipts. Climate warming is migrating heat-stressed
# southern dairy operations north at an estimated 2%/yr climate-driven advantage
# (USDA ERS 2023 heat-stress livestock report; Mauger et al. 2015 for PNW).
# Over a 15-year horizon to 2040: $30B × (1.02^15 - 1) ≈ $30B × 0.35 = $10.5B
# cumulative net growth, equivalent to ~$0.7B/yr ongoing. For the full 2040 stock
# opportunity (new infrastructure + dairy herd expansion), use the lump-sum:
# $30B × 0.30 = $9B. This is conservative (excludes beef/pork co-migration).
DAIRY_LIVESTOCK_OPPORTUNITY_B: float = 9.0
DAIRY_STATES = {'27', '55', '50', '36', '26', '53', '41', '16'}  # WI,MN,VT,NY,MI,WA,OR,ID


def _fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (f - 32.0) * 5.0 / 9.0


def compute_income_gain(
    county_fips: str,
    yield_projections: pd.DataFrame,
    yield_current: pd.DataFrame,
    scenario: str = 'RCP45'
) -> dict:
    """Compute projected income gain for a northern county (yield component only).

    This captures the benefit of higher yields on existing harvested acreage.
    income_gain = (projected_yield - current_yield) × price × current_acres

    Args:
        county_fips: 5-digit FIPS code.
        yield_projections: Projected yields under scenario (2035-2045 avg used as 2040 estimate).
        yield_current: Current (2019-2023 avg) yields.
        scenario: Climate scenario label for output tagging.

    Returns:
        Dict with annual_income_gain, income_gain_per_acre, total_acres, and per-crop breakdown.
    """
    proj = yield_projections[yield_projections['fips'] == county_fips]
    curr = yield_current[yield_current['fips'] == county_fips]

    if proj.empty:
        return {
            'fips': county_fips, 'annual_income_gain': 0.0,
            'income_gain_per_acre': 0.0, 'total_acres': 0.0,
        }

    total_gain = 0.0
    total_acres = 0.0

    for crop in proj['crop'].unique():
        crop_proj = proj[proj['crop'] == crop]
        crop_curr = curr[curr['crop'] == crop] if not curr.empty else pd.DataFrame()

        projected_yield = float(crop_proj['yield_projected'].mean())
        if pd.isna(projected_yield):
            continue
        current_yield = float(crop_curr['yield_bu_acre'].mean()) if not crop_curr.empty else 0.0
        if pd.isna(current_yield):
            current_yield = 0.0
        acres = (
            float(crop_curr['acres_harvested'].mean())
            if not crop_curr.empty and 'acres_harvested' in crop_curr.columns
            else 0.0
        )
        if pd.isna(acres):
            acres = 0.0

        price = COMMODITY_PRICES.get(crop, 5.0)
        gain = (projected_yield - current_yield) * price * acres
        total_gain += max(gain, 0.0)
        total_acres += acres

    gain_per_acre = total_gain / total_acres if total_acres > 0 else 0.0

    return {
        'fips': county_fips,
        'scenario': scenario,
        'annual_income_gain': float(total_gain),
        'income_gain_per_acre': float(gain_per_acre),
        'total_acres': float(total_acres),
    }


def compute_acreage_expansion(
    county_fips: str,
    yield_projections: pd.DataFrame,
    yield_current: pd.DataFrame,
    farm_ops: pd.DataFrame,
) -> dict:
    """Compute acreage expansion income potential for a northern county.

    Warming makes currently marginal/idle cropland viable for major crops.
    Approach:
        total_farmland = Census of Ag 'FARM OPERATIONS - ACRES OPERATED' (most recent year)
        cropland_acres = total_farmland × state_cropland_fraction
            — corrects for the fact that 'ACRES OPERATED' includes rangeland/pasture
        current_utilization = harvested_acres / cropland_acres  (capped at 1)
        expandable_acres = cropland_acres × (1 - utilization)
        expansion_income = expandable_acres × projected_yield × price
            — for the best (highest-value) crop with positive climate signal

    Args:
        county_fips: 5-digit FIPS code.
        yield_projections: Projected 2040 yields.
        yield_current: Current (2019-2023) yields.
        farm_ops: NASS farm operations data (total acres operated, Census of Ag).

    Returns:
        Dict with expansion_income, expandable_acres, utilization_rate.
    """
    state_code = str(county_fips).zfill(5)[:2]
    cropland_frac = STATE_CROPLAND_FRACTION.get(state_code, 0.50)

    # Total farmland from Census of Ag
    county_ops = farm_ops[
        (farm_ops['fips'] == county_fips) &
