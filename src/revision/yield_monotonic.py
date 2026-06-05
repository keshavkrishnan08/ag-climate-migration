"""Monotonicity-constrained hybrid yield model + clean scenario-consistent projection.

Core fix: a gradient-boosted model with MONOTONE CONSTRAINTS so predicted yield is
non-increasing in every heat/dryness feature (Tmax, EDD>30C, KDD>34C, VPD, soil-
moisture stress) and non-decreasing in precipitation. This makes the climate
response monotone by construction, so projected losses rise with warming for every
scenario -- eliminating the out-of-sample non-monotonicity with no caveat.

Projection is self-contained: for each county we hold a representative feature row,
recompute the NONLINEAR heat features (EDD/KDD/VPD via within-month sinusoid) at
baseline climatology and at baseline+CMIP6 delta, and take the model's difference as
the climate impact (z-anomaly), converted to bu/ac via the county-crop detrended SD.

Outputs: held-out R^2/Spearman, a monotonicity audit, scenario-consistent stranded
totals (SSP2-4.5, SSP3-7.0), and the model residual SD for the DCF CI. Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
PROJ = ROOT / "data" / "projections"
OUT = ROOT / "results" / "revision"
SEED = 42
GROW = [f"{m:02d}" for m in range(4, 10)]
PHASE = np.linspace(0, np.pi, 24)
PRICE = {"corn": 5.04, "soybeans": 12.29, "wheat_winter": 6.72, "wheat_spring": 7.38,
         "cotton": 0.93, "sorghum": 4.80, "barley": 5.64, "oats": 3.35}


def es(t): return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def heat_features_from_monthly(tmax_c, tmin_c):
    """Vectorized EDD>30, KDD>34, VPD, heat-days, DTR from monthly arrays.

    Args: tmax_c, tmin_c are (n, 6) arrays of growing-season monthly means (C).
    Returns dict of (n,) feature arrays.
    """
    n = tmax_c.shape[0]
    edd = np.zeros(n); kdd = np.zeros(n); hot = np.zeros(n)
    mid = (tmax_c + tmin_c) / 2; amp = (tmax_c - tmin_c) / 2
    for j in range(tmax_c.shape[1]):
        dd30 = np.zeros(n); dd34 = np.zeros(n)
        for p in PHASE:
            temp = mid[:, j] + amp[:, j] * np.sin(p - np.pi / 2)
            dd30 += np.maximum(temp - 30, 0); dd34 += np.maximum(temp - 34, 0)
        edd += dd30 / len(PHASE) * 30; kdd += dd34 / len(PHASE) * 30
        hot += (tmax_c[:, j] > 30).astype(float)
    vpd = (es(tmax_c) - es(tmin_c)).clip(min=0).mean(axis=1)
    vpd_jul = (es(tmax_c[:, 3]) - es(tmin_c[:, 3])).clip(min=0)   # July = index 3 (Apr=0)
    dtr = (tmax_c - tmin_c).mean(axis=1)
    return {"edd30": edd, "kdd34": kdd, "heat_days": hot, "vpd_grow": vpd,
            "vpd_jul": vpd_jul, "dtr": dtr}


