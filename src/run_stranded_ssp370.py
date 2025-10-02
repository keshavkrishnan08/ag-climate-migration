"""Compute stranded agricultural assets under SSP3-7.0.

Runs the same conservative (ML only) and central (ML + SR + indirect) methods
as 06_stranded.py but using SSP370 yield and climate projections.
Reports results and compares against the SSP245 baseline.
"""

import sys
import json
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
OUTPUT_DIR = RESULTS_DIR / 'stranded_assets'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

SCENARIO = 'SSP370'

# ── Inline helper functions (mirrors 06_stranded.py) ─────────────────────────
