"""Phase 4: Future projections — CMIP6 scenarios 2025-2050.

Projects county-level crop yields using:
1. Trained yield model (Phase 3A)
2. Crop switching models (Phase 3B)
3. CMIP6 climate projections downscaled to county level (pre-computed)

Primary scenario: SSP2-4.5 (~RCP 4.5), +1.4-1.8°C by 2050.
GCM ensemble: 5 CMIP6 models, median + 10-90th percentile uncertainty.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
import yaml
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
PROJECTIONS_DIR = PROJECT_ROOT / 'data' / 'projections'
RESULTS_DIR = PROJECT_ROOT / 'results'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)
