"""Phase 3A: Core yield trend model — LightGBM.

Predicts county-crop yield as a function of climate and technology features.
This is the engine of all projections.

Architecture (PRD Section 5.1):
    - LightGBM gradient boosted trees
    - Outcome: detrended yield anomaly (z-score)
    - After prediction: re-add projected technology trend
    - Final projected yield = tech_component + climate_component

Cross-Validation:
    STRICT TEMPORAL ROLLING CV — never shuffle
    Fold 1: train 1950-1985, val 1986-1990
    Fold 2: train 1950-1990, val 1991-1995
    Fold 3: train 1950-1995, val 1996-2000
    Fold 4: train 1950-2000, val 2001-2005
    Fold 5: train 1950-2005, val 2006-2010
    Final test: train 1950-2012, test 2013-2023

Performance thresholds (anomaly model — z-scored target):
    Spearman ≥ 0.35 overall (Schlenker & Roberts 2009 benchmark)
    Spearman ≥ 0.30 per crop (cotton excluded — driven by non-climate factors)
    R² ≥ 0.10 on z-scored anomalies (expected range ~0.15-0.25)
    Note: raw-yield R² thresholds (corn≥0.72 etc.) do NOT apply here —
    z-scored anomaly R² ~0.18 is literature-appropriate and not a model deficiency.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from scipy import stats
from loguru import logger
import yaml
import json
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from utils.validation import temporal_rolling_cv, check_no_future_leakage, compute_performance_metrics
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'

