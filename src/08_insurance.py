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
