"""Phase 3D: Retrain yield model with compound drought interaction features.

Reviewer fix: the base model under-predicts the 2012 drought because it misses
the multiplicative penalty when heat AND moisture stress occur simultaneously.
This script adds compound interaction features and retrains with depth=8 to
let the tree structure capture the threshold behavior.

New features added:
    heat_x_drought     = tmax_july_c_anomaly * (-pdsi_growing_anomaly)
    heat_x_precip      = tmax_july_c_anomaly * (-precip_growing_anomaly)
    extreme_compound   = (tmax_anom > 1°C) AND (pdsi_anom < -1) binary flag
    tmax_july_sq       = tmax_july_c_anomaly² (quadratic heat threshold)
    precip_deficit     = max(0, county_baseline_precip - actual_precip)
    tmax_peak_c        = max(tmax Jun/Jul/Aug) in Celsius (from monthly data)
    precip_jja         = total Jun+Jul+Aug precipitation
    pdsi_peak_drought  = min(PDSI Jun/Jul/Aug) — worst summer drought month
    edd_months_c       = months where tmax > 33.5°C (Schlenker & Roberts threshold)
    edd_x_pdsi         = edd_months_c * (-pdsi_peak_drought) compound signal
    + anomaly versions of the monthly features (county mean subtracted)

Temporal split (matches config.yaml val_end=2012):
    Train:  years <= 2012
    Test:   2013-2023

Saves:
    results/yield_model_v2.pkl   — new primary model
    results/yield_model_v2_metrics.json
    results/feature_importance_v2.csv
"""

import os
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from loguru import logger
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from utils.validation import (
    temporal_rolling_cv,
    check_no_future_leakage,
    compute_performance_metrics,
)

DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
RESULTS_DIR = PROJECT_ROOT / 'results'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

RANDOM_SEED = CONFIG['yield_model']['random_seed']

# Schlenker & Roberts (2009) EDD threshold for corn (°C)
EDD_THRESHOLD_C = 33.5

# Growing-season months: May–September
GROWING_MONTHS = ['05', '06', '07', '08', '09']
# Critical pollination months: June–August
PEAK_MONTHS = ['06', '07', '08']


def load_monthly_features() -> pd.DataFrame:
    """Load and derive monthly climate features for compound drought detection.

    Converts nClimDiv temperatures from Fahrenheit to Celsius, then computes:
    - tmax_peak_c:      max(Jun, Jul, Aug) Tmax in Celsius
    - precip_jja:       total Jun+Jul+Aug precipitation
    - pdsi_peak_drought: min(Jun,Jul,Aug PDSI) — most severe monthly drought
    - edd_months_c:     count of months with Tmax > 33.5°C (EDD threshold)
