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
        (farm_ops['short_desc'].str.contains('ACRES OPERATED', na=False))
    ]

    # Use most recent Census year; convert total farm acres to cropland acres
    total_cropland = 0.0
    if not county_ops.empty:
        recent = county_ops.sort_values('year', ascending=False)
        total_farmland = float(recent['value'].iloc[0])
        total_cropland = total_farmland * cropland_frac

    # Harvested acres (current): mean per crop across years, then sum crops.
    # Cannot do a raw .sum() because yield_current has one row per fips-year-crop.
    curr = yield_current[yield_current['fips'] == county_fips]
    if not curr.empty:
        current_harvested = float(
            curr.groupby('crop')['acres_harvested'].mean().sum()
        )
    else:
        current_harvested = 0.0

    # If cropland unknown or implausibly small, infer from harvested acres
    if total_cropland <= 0 or total_cropland < current_harvested:
        total_cropland = current_harvested / DEFAULT_UTILIZATION_RATE if current_harvested > 0 else 0.0

    if total_cropland <= 0:
        return {
            'fips': county_fips,
            'expandable_acres': 0.0,
            'utilization_rate': DEFAULT_UTILIZATION_RATE,
            'expansion_income': 0.0,
            'expansion_crop': None,
        }

    utilization = min(current_harvested / total_cropland, 1.0) if total_cropland > 0 else DEFAULT_UTILIZATION_RATE
    expandable_acres = total_cropland * (1.0 - utilization)

    if expandable_acres <= 0:
        return {
            'fips': county_fips,
            'expandable_acres': 0.0,
            'utilization_rate': float(utilization),
            'expansion_income': 0.0,
            'expansion_crop': None,
        }

    # Find the best projected crop for expansion acres.
    # For expansion on PREVIOUSLY IDLE land, the relevant baseline is zero production;
    # any positive projected yield is a gain. We therefore do NOT filter on
    # climate_impact_bu (which measures delta vs existing production, not vs zero).
    # Instead, we require projected_yield > a minimum viability threshold to exclude
    # counties where the model projects near-zero yields (effectively "won't grow there").
    #
    # Minimum viable yield thresholds (bu/acre) — roughly 50% of current northern state
    # averages, meaning the new land would produce at least half as well as existing fields:
    MIN_VIABLE_YIELD = {
        'corn': 60.0, 'soybeans': 20.0, 'wheat_winter': 30.0,
        'wheat_spring': 20.0, 'barley': 30.0, 'oats': 25.0,
        'sorghum': 40.0,
    }

    proj = yield_projections[yield_projections['fips'] == county_fips]
    if proj.empty:
        return {
            'fips': county_fips,
            'expandable_acres': float(expandable_acres),
            'utilization_rate': float(utilization),
            'expansion_income': 0.0,
            'expansion_crop': None,
        }

    best_income = 0.0
    best_crop = None

    for crop in proj['crop'].unique():
        cp = proj[proj['crop'] == crop]
        projected_yield = float(cp['yield_projected'].mean())

        # Require minimum viable yield — excludes areas where the crop physically won't grow
        min_viable = MIN_VIABLE_YIELD.get(crop, 20.0)
        if projected_yield < min_viable:
            continue

        price = COMMODITY_PRICES.get(crop, 5.0)
        income = expandable_acres * projected_yield * price
        if income > best_income:
            best_income = income
            best_crop = crop

    return {
        'fips': county_fips,
        'expandable_acres': float(expandable_acres),
        'utilization_rate': float(utilization),
        'expansion_income': float(best_income),
        'expansion_crop': best_crop,
    }


def compute_gdd_base10(
    county_fips: str,
    climate_monthly: pd.DataFrame,
    climate_proj: pd.DataFrame,
    target_year: int = 2040,
) -> float:
    """Estimate growing-degree days (GDD, base 10°C) for a county in the target year.

    Uses projected delta-T on top of the historical (2019-2023) baseline.
    Growing season = April through September (months 4-9).

    Monthly GDD contribution:
        gdd_month = max(0, T_avg_C - 10) × days_in_month
    T_avg_C = (tmax_C + tmin_C) / 2

    Args:
        county_fips: 5-digit FIPS code.
        climate_monthly: Historical monthly climate (°F tmax/tmin columns).
        climate_proj: County climate projections with delta_tmax_growing.
        target_year: Projection year for climate delta.

    Returns:
        Projected annual GDD (base 10°C) for the growing season.
    """
    DAYS = {4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30}  # Apr-Sep

    hist = climate_monthly[
        (climate_monthly['fips'] == county_fips) &
        (climate_monthly['year'].between(2019, 2023))
    ]

    if hist.empty:
        return 0.0

    # Historical baseline GDD
    gdd_hist = 0.0
    for month, days in DAYS.items():
        m_str = f'm{month:02d}'
        tmax_col = f'tmax_{m_str}'
        tmin_col = f'tmin_{m_str}'
        if tmax_col not in hist.columns or tmin_col not in hist.columns:
            continue
        tmax_f = hist[tmax_col].mean()
        tmin_f = hist[tmin_col].mean()
        tmax_c = _fahrenheit_to_celsius(tmax_f)
        tmin_c = _fahrenheit_to_celsius(tmin_f)
        t_avg_c = (tmax_c + tmin_c) / 2.0
        gdd_hist += max(0.0, t_avg_c - 10.0) * days

    # Projected warming delta for target year
    proj_row = climate_proj[
        (climate_proj['fips'] == county_fips) &
        (climate_proj['year'] == target_year)
    ]
    delta_c = 0.0
    if not proj_row.empty and 'delta_tmax_growing' in proj_row.columns:
        delta_f = float(proj_row['delta_tmax_growing'].iloc[0])
        # delta_tmax_growing is stored in °F (raw PRISM delta)
        delta_c = delta_f * 5.0 / 9.0

    # Apply delta uniformly across growing months
    growing_months = len(DAYS)
    total_days = sum(DAYS.values())  # 183 days Apr-Sep
    gdd_projected = gdd_hist + delta_c * total_days

    return max(0.0, gdd_projected)


def compute_crop_upgrade(
    county_fips: str,
    yield_projections: pd.DataFrame,
    yield_current: pd.DataFrame,
    gdd_projected: float,
) -> dict:
    """Compute crop-upgrade income for counties growing low-value crops.

    If warming brings GDD above corn/soy thresholds, counties currently growing
    oats or barley can switch to higher-value crops. The upgrade premium captures
    the revenue gap between what could be grown vs what's currently grown.

    upgrade_income = current_low_value_acres × (high_value_revenue_per_acre - low_value_revenue_per_acre)

    Args:
        county_fips: 5-digit FIPS code.
        yield_projections: Projected 2040 yields.
        yield_current: Current (2019-2023) yields.
        gdd_projected: Projected GDD base 10°C for 2040.

    Returns:
        Dict with upgrade_income, upgrade_acres, target_crop, and gdd info.
    """
    curr = yield_current[yield_current['fips'] == county_fips]
    if curr.empty:
        return {
            'fips': county_fips, 'upgrade_income': 0.0,
            'upgrade_acres': 0.0, 'target_crop': None,
            'gdd_projected': float(gdd_projected),
        }

    # Identify low-value acreage
    low_val_curr = curr[curr['crop'].isin(LOW_VALUE_CROPS)]
    if low_val_curr.empty:
        return {
            'fips': county_fips, 'upgrade_income': 0.0,
            'upgrade_acres': 0.0, 'target_crop': None,
            'gdd_projected': float(gdd_projected),
        }

    # Determine viable upgrade crop based on GDD
    if gdd_projected >= GDD_CORN_MIN:
        target_crop = 'corn'
    elif gdd_projected >= GDD_SOY_MIN:
        target_crop = 'soybeans'
    else:
        # GDD not sufficient for high-value crop upgrade
        return {
            'fips': county_fips, 'upgrade_income': 0.0,
            'upgrade_acres': 0.0, 'target_crop': None,
            'gdd_projected': float(gdd_projected),
        }

    proj = yield_projections[yield_projections['fips'] == county_fips]
    target_proj = proj[proj['crop'] == target_crop] if not proj.empty else pd.DataFrame()
    target_yield = float(target_proj['yield_projected'].mean()) if not target_proj.empty else 0.0
    target_price = COMMODITY_PRICES[target_crop]

    total_upgrade_income = 0.0
    total_upgrade_acres = 0.0

    for crop in LOW_VALUE_CROPS:
        crop_curr = low_val_curr[low_val_curr['crop'] == crop]
        if crop_curr.empty:
            continue

        low_acres = float(crop_curr['acres_harvested'].mean())
        low_yield = float(crop_curr['yield_bu_acre'].mean())
        low_price = COMMODITY_PRICES.get(crop, 4.0)

        low_revenue_per_acre = low_yield * low_price
        high_revenue_per_acre = target_yield * target_price
        premium = max(0.0, high_revenue_per_acre - low_revenue_per_acre)

        total_upgrade_income += premium * low_acres
        total_upgrade_acres += low_acres

    return {
        'fips': county_fips,
        'upgrade_income': float(total_upgrade_income),
        'upgrade_acres': float(total_upgrade_acres),
        'target_crop': target_crop,
        'gdd_projected': float(gdd_projected),
    }


def compute_infrastructure_capacity(
    county_fips: str,
    elevator_data: pd.DataFrame,
    projected_production: float
) -> dict:
    """Compute infrastructure capacity ratio for a county.

    Infrastructure capacity ratio:
        = min(grain_elevator_capacity, rail_capacity, processing_capacity)
        / projected_production_2040
    If < 1: infrastructure is the binding constraint.

    Args:
        county_fips: 5-digit FIPS code.
        elevator_data: GIPSA grain elevator data.
        projected_production: Projected total production (bushels).

    Returns:
        Dict with capacity metrics.
    """
    county_elevators = elevator_data[elevator_data['fips'] == county_fips] if not elevator_data.empty else pd.DataFrame()

    if not county_elevators.empty and 'storage_capacity_bushels' in county_elevators.columns:
        elevator_capacity = county_elevators['storage_capacity_bushels'].sum()
    else:
        elevator_capacity = 0.0

    if projected_production > 0:
        capacity_ratio = min(elevator_capacity / projected_production, 1.0)
    else:
        capacity_ratio = 1.0

    uncaptured = 1.0 - capacity_ratio

    investment_per_bushel = 5.0
    if projected_production > elevator_capacity:
        investment_needed = (projected_production - elevator_capacity) * investment_per_bushel
    else:
        investment_needed = 0.0

    return {
        'fips': county_fips,
        'elevator_capacity': float(elevator_capacity),
        'projected_production': float(projected_production),
        'infrastructure_capacity_ratio': float(capacity_ratio),
        'uncaptured_fraction': float(uncaptured),
        'infrastructure_investment_needed': float(investment_needed),
    }


def compute_northern_opportunity(
    county_fips: str,
    yield_projections: pd.DataFrame,
    yield_current: pd.DataFrame,
    elevator_data: pd.DataFrame,
    farm_ops: pd.DataFrame,
    climate_monthly: pd.DataFrame,
    climate_proj: pd.DataFrame,
    scenario: str = 'RCP45',
    target_year: int = 2040,
) -> dict:
    """Full northern opportunity decomposition for one county.

    Aggregates three components:
        1. Yield gain on existing harvested acres.
        2. Acreage expansion into idle/marginal farmland.
        3. Crop-upgrade premium for oat/barley counties crossing GDD thresholds.

    Also computes:
        - Infrastructure investment needed (expansion_acres × $500/acre USDA estimate).
        - Legacy infrastructure capacity ratio (grain elevator basis).

    Args:
        county_fips: 5-digit FIPS code.
        yield_projections: Projected yields (all years; 2035-2045 window used for 2040 avg).
        yield_current: Current (2019-2023) yields.
        elevator_data: Grain elevator data.
        farm_ops: NASS farm operations (Census of Ag total acres).
        climate_monthly: Historical monthly climate °F.
        climate_proj: County climate projections with delta_tmax_growing.
        scenario: Climate scenario label.
        target_year: Projection year (default 2040).

    Returns:
        Dict with full opportunity breakdown.
    """
    # Use 2035-2045 window as the 2040 estimate
    proj_window = yield_projections[
        yield_projections['year'].between(target_year - 5, target_year + 5)
    ]

    # 1. Yield gain component
    income = compute_income_gain(county_fips, proj_window, yield_current, scenario)

    # 2. Acreage expansion component
    expansion = compute_acreage_expansion(county_fips, proj_window, yield_current, farm_ops)

    # 3. GDD → crop upgrade component
    gdd = compute_gdd_base10(county_fips, climate_monthly, climate_proj, target_year)
    upgrade = compute_crop_upgrade(county_fips, proj_window, yield_current, gdd)

    # 4. Infrastructure capacity (grain elevator basis)
    proj_production = 0.0
    county_proj = proj_window[proj_window['fips'] == county_fips]
    if not county_proj.empty and 'yield_projected' in county_proj.columns:
        for _, row in county_proj.iterrows():
            proj_production += row.get('yield_projected', 0.0) * row.get('acres_harvested', 0.0)
    infra = compute_infrastructure_capacity(county_fips, elevator_data, proj_production)

    # 5. Infrastructure investment for expansion (USDA $500/acre standard)
    expansion_infra_investment = expansion['expandable_acres'] * INFRA_COST_PER_ACRE

    # Totals — guard against NaN from missing NASS data
    total_opportunity = (
        float(income.get('annual_income_gain') or 0.0)
        + float(expansion.get('expansion_income') or 0.0)
        + float(upgrade.get('upgrade_income') or 0.0)
    )

    return {
        'fips': county_fips,
        'scenario': scenario,
        # Component 1: yield gains
        'yield_gain_income': income['annual_income_gain'],
        'income_gain_per_acre': income['income_gain_per_acre'],
        'current_harvested_acres': income['total_acres'],
        # Component 2: acreage expansion
        'expandable_acres': expansion['expandable_acres'],
        'utilization_rate': expansion['utilization_rate'],
        'expansion_income': expansion['expansion_income'],
        'expansion_crop': expansion['expansion_crop'],
        # Component 3: crop upgrade
        'upgrade_income': upgrade['upgrade_income'],
        'upgrade_acres': upgrade['upgrade_acres'],
        'target_crop': upgrade['target_crop'],
        'gdd_projected': upgrade['gdd_projected'],
        # Legacy infrastructure (grain elevator basis)
        'elevator_capacity': infra['elevator_capacity'],
        'projected_production': infra['projected_production'],
        'infrastructure_capacity_ratio': infra['infrastructure_capacity_ratio'],
        'uncaptured_fraction': infra['uncaptured_fraction'],
        'infrastructure_investment_needed': infra['infrastructure_investment_needed'],
        # Expansion infrastructure investment (USDA $500/acre)
        'expansion_infra_investment': float(expansion_infra_investment),
        # Aggregate
        'total_annual_opportunity': float(total_opportunity),
        'annual_income_gain': float(total_opportunity),  # backward-compat alias
        'annual_opportunity_2023USD': float(total_opportunity),
        'opportunity_with_gap': float(total_opportunity * infra['uncaptured_fraction']),
    }


def identify_opportunity_counties(
    yield_projections: pd.DataFrame,
    yield_current: pd.DataFrame,
    elevator_data: pd.DataFrame,
    farm_ops: pd.DataFrame,
    climate_monthly: pd.DataFrame,
    climate_proj: pd.DataFrame,
    income_threshold_per_acre: float = 5.0,
    scenario: str = 'RCP45',
    target_year: int = 2040,
) -> pd.DataFrame:
    """Identify and rank top northern opportunity counties.

    A county qualifies if ANY of these hold:
        - income_gain_per_acre > $5/acre (yield component), OR
        - expansion_income > 0 (idle farmland + positive climate signal), OR
        - upgrade_income > 0 (crop upgrade GDD threshold met)

    Args:
        yield_projections: Projected yields (all years).
        yield_current: Current (2019-2023) yields.
        elevator_data: Grain elevator data.
        farm_ops: NASS farm operations (Census of Ag total acres).
        climate_monthly: Historical monthly climate °F.
        climate_proj: County climate projections.
        income_threshold_per_acre: Minimum yield-gain per acre to qualify (yield component gate).
        scenario: Climate scenario label.
        target_year: Projection year.

    Returns:
        DataFrame of opportunity counties ranked by total_annual_opportunity descending.
    """
    logger.info("Identifying northern opportunity counties (3-component model)...")

    results = []
    all_fips = yield_projections['fips'].unique() if not yield_projections.empty else []

    for fips in all_fips:
        state_code = str(fips).zfill(5)[:2]
        if state_code not in NORTHERN_STATES:
            continue

        result = compute_northern_opportunity(
            fips, yield_projections, yield_current, elevator_data,
            farm_ops, climate_monthly, climate_proj, scenario, target_year,
        )
        result['state'] = NORTHERN_STATES.get(state_code, 'Unknown')
        results.append(result)

    df = pd.DataFrame(results)

    if df.empty:
        return pd.DataFrame()

    # Filter: county qualifies if any component is positive
    opportunity = df[
        (df['income_gain_per_acre'] > income_threshold_per_acre) |
        (df['expansion_income'] > 0) |
        (df['upgrade_income'] > 0)
    ].sort_values('total_annual_opportunity', ascending=False)

    logger.info(f"\nNORTHERN OPPORTUNITY FRONTIER ({scenario}, target year {target_year}):")
    logger.info(f"  Northern counties analyzed: {len(df)}")
    logger.info(f"  Counties meeting opportunity criteria: {len(opportunity)}")

    if not opportunity.empty:
        yield_gain_B = opportunity['yield_gain_income'].sum() / 1e9
        expansion_B = opportunity['expansion_income'].sum() / 1e9
        upgrade_B = opportunity['upgrade_income'].sum() / 1e9
        total_B = opportunity['total_annual_opportunity'].sum() / 1e9
        avg_capacity = opportunity['infrastructure_capacity_ratio'].mean()
        total_elev_inv_B = opportunity['infrastructure_investment_needed'].sum() / 1e9
        total_exp_inv_B = opportunity['expansion_infra_investment'].sum() / 1e9

        logger.info(f"  --- Opportunity Components ---")
        logger.info(f"  1. Yield gains (existing acres):    ${yield_gain_B:.1f}B/yr")
        logger.info(f"  2. Acreage expansion (idle land):   ${expansion_B:.1f}B/yr")
        logger.info(f"  3. Crop upgrading (oats/barley→):   ${upgrade_B:.1f}B/yr")
        logger.info(f"  TOTAL annual opportunity:           ${total_B:.1f}B/yr")
        logger.info(f"  --- Infrastructure ---")
        logger.info(f"  Avg infrastructure capacity ratio:  {avg_capacity:.0%}")
        logger.info(f"  Elevator investment needed:         ${total_elev_inv_B:.1f}B")
        logger.info(f"  Expansion infrastructure needed:    ${total_exp_inv_B:.1f}B")

    return opportunity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_frontier_analysis() -> dict:
    """Execute northern opportunity frontier analysis.

    Loads all required data, runs the 3-component opportunity model across
    all northern counties, and writes results plus updates headline_numbers_preliminary.json.

    Returns:
        Dict with opportunity county DataFrame and summary statistics.
    """
    logger.info("=" * 60)
    logger.info("PHASE 5D: NORTHERN OPPORTUNITY FRONTIER (3-COMPONENT MODEL)")
    logger.info("=" * 60)

    output_dir = RESULTS_DIR / 'frontier'
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = 'SSP245'
    target_year = 2040

    # ----- Load yield projections -----
    proj_path = PROJECTIONS_DIR / f'yield_projections_{scenario}.parquet'
    if proj_path.exists():
        all_proj = pd.read_parquet(proj_path)
        northern_fips_prefix = set(NORTHERN_STATES.keys())
        yield_proj = all_proj[
            all_proj['fips'].str[:2].isin(northern_fips_prefix)
        ].copy()
        logger.info(f"Northern yield projections: {len(yield_proj)} rows, "
                    f"{yield_proj['fips'].nunique()} counties")
    else:
        logger.error("No yield projections found — aborting frontier analysis")
        return {'opportunity_counties': pd.DataFrame()}

    # ----- Load NASS yields (current baseline) -----
    current_path = DATA_RAW / 'nass' / 'nass_county_yields.parquet'
    if current_path.exists():
        nass_raw = pd.read_parquet(
