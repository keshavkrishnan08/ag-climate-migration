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
