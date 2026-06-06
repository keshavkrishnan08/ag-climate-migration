"""Yield v6: maximize anomaly R^2 (skill model) -- per-crop, unconstrained, with
soil, latitude, and a same-year regional (state) spatial-lag anomaly.

The spatial lag (mean yield anomaly of OTHER counties in the same state and year)
captures the regional weather signal an individual county's aggregates miss; it is
a legitimate predictor (neighbouring observations are available). This is the
reported model-skill metric. Projection monotonicity is handled separately by the
process-based damage function, so this model is unconstrained. Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_model_v3_features import build_modern_features
from yield_v4_morefeatures import extra_features
from yield_v5_percrop import latitude
DATA_PROCESSED = ROOT / "data" / "processed"
