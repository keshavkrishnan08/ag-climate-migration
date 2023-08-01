"""Phase 3C: Quantile Regression Forest for Extreme Event Tail Risk.

The mean yield model underpredicts extreme events — the 2012 drought anomaly
prediction of -0.11 is far too mild vs. the observed mean of -0.94 across
corn counties. Tail risk matters for stranded asset estimates, which depend
on correctly capturing downside scenarios.

This script:
  1. Trains a LightGBM quantile model at alpha=0.10 (Q10) using same features
     and temporal split as the existing mean model.
  2. Diagnoses the 2012 drought: Q10 prediction should be much more negative
     than the mean model's -0.11.
  3. Computes a tail risk premium per county (mean - Q10 gap) and adds it as
     an additional stranded asset component.

Temporal split (per CLAUDE.md):
    train  : year <= 2009
    val    : 2010-2016  (used as final training cutoff for test evaluation)
    test   : 2017-2023

Output:
    models/yield/yield_model_q10.pkl
    results/quantile/q10_metrics.json
    results/quantile/drought_2012_q10.csv
    results/quantile/tail_risk_stranded.parquet
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
PROJECTIONS_DIR = PROJECT_ROOT / 'data' / 'projections'
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
MODELS_DIR = PROJECT_ROOT / 'models' / 'yield'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'quantile'

with open(PROJECT_ROOT / 'config.yaml') as f:
