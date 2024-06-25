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
