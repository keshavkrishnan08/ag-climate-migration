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
