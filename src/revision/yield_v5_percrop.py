"""Yield v5: per-crop monotone models with soil, latitude, and autoregressive
features -- raises anomaly R^2 without dropping any crop.

Additions over v4:
  * PER-CROP models (climate-yield response differs by crop; pooled dilutes it).
  * Soil quality proxy (NCCPI-style: county max historical yield / national max).
  * County latitude (centroid, USDA gazetteer).
  * Autoregressive prior-year own anomaly (AR1) -- persistence from soil moisture
    carryover and management; known at planting, set to 0 under projection.
  * Monotone climate constraints retained (heat/dry -1, precip +1) so the model
    stays physically monotone for projection.
Reports per-crop and overall held-out (2013-2023) R^2/Spearman. Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_model_v3_features import build_modern_features
from yield_v4_morefeatures import extra_features
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "revision"
SEED = 42

CLIM_SIGN = {  # anomaly feature -> monotone sign
    "tmax_july_c_anomaly": -1, "vpd_growing_anom": -1, "vpd_july_anom": -1,
    "edd30_growing_anom": -1, "heat_days_proxy_anom": -1, "sm_stress_anom": -1,
    "sm_stress_july_anom": -1, "vpd_x_sm_anom": -1, "kdd34_growing_anom": -1,
    "dtr_growing_anom": -1, "precip_jul_anom": +1, "precip_aug_anom": +1,
    "precip_growing_anomaly": +1,
}
NONCLIM = ["yield_trend_slope_15yr", "yield_trend_intercept", "log_population",
           "log_median_income", "poverty_rate", "switching_rate_proxy",
           "switching_rate_5yr", "latitude", "nccpi", "ar1"]


def latitude():
