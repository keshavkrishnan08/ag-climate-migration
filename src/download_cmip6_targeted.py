"""Targeted CMIP6 download — only what the paper needs.

Downloads 5 diverse GCMs x ssp245 x 3 vars x milestone years.
Then linearly interpolates annual values for the projection pipeline.

Usage:
    python src/download_cmip6_targeted.py
"""

import sys
from pathlib import Path

# Reuse the main download machinery
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_cmip6 import process_one_file, CMIP6_DIR, VARIABLES, MODEL_VARIANTS
from loguru import logger
import time
import pandas as pd
import numpy as np
import os
