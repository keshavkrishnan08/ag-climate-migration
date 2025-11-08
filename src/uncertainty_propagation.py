"""Monte Carlo uncertainty propagation for stranded asset DCF estimate.

Propagates yield model uncertainty (R²=0.21) through the DCF stranded asset
computation using 1,000 Monte Carlo draws from the residual distribution.

Outputs:
    results/stranded_assets/uncertainty_propagation.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
RESULTS_DIR = PROJECT_ROOT / 'results'
