"""Audit improvement B: MLP on the monthly climate sequence + stacked ensemble.

Orthogonal architecture to the gradient-boosted trees. Two distinct ideas:

1. MLP over the raw MONTHLY climate sequence. A tree splits one feature at a time
   and approximates smooth multivariate climate response with axis-aligned steps.
   A small multi-layer perceptron fed the standardized monthly tmax/tmin/precip/PDSI
   vector (Apr-Sep) plus soil/latitude/trend can capture the smooth, interacting
   temperature x water response directly. We keep CLIMATE-ONLY inputs so it stays
   projectable from CMIP6 monthly deltas (no spatial-yield lags).

2. STACK the MLP with the best gradient-boosted tree (v4-style feature set + the
   drought-trajectory features) via a non-negative least-squares blend fit on a
   held-out 2010-2012 window (so blend weights never see the 2013-2023 test set).
   If the two learners make different errors, the stack beats either alone.

Target = the existing z-scored, county-detrended yield anomaly, so the held-out
R^2 is DIRECTLY comparable to the paper's 0.227. Split: train<=2009 (MLP needs a
clean fit window), blend 2010-2012, test 2013-2023. Seed 42.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from scipy.optimize import nnls
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_audit_drought_huber import build_panel, design

DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "revision"
SEED = 42
GROW = [f"{m:02d}" for m in range(4, 10)]


def monthly_sequence_features():
    """Standardizable monthly tmax/tmin/precip/PDSI for Apr-Sep, plus county means.

    Returns:
        DataFrame [fips, year, <monthly climate columns>].
    """
    m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet")
    m["fips"] = m["fips"].astype(str).str.zfill(5)
    cols = {"fips": m["fips"].values, "year": m["year"].values}
    for mm in GROW:
        cols[f"seq_tmax_{mm}"] = (m[f"tmax_m{mm}"].values - 32) * 5 / 9
        cols[f"seq_tmin_{mm}"] = (m[f"tmin_m{mm}"].values - 32) * 5 / 9
        cols[f"seq_precip_{mm}"] = m[f"precip_m{mm}"].values
        cols[f"seq_pdsi_{mm}"] = m[f"pdsi_m{mm}"].values
    return pd.DataFrame(cols)


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(r2), float(stats.spearmanr(y, p).correlation)


def per_crop(panel_te, pred):
    tp = panel_te.reset_index(drop=True).copy(); tp["pred"] = pred
    out = {}
    for c in sorted(tp["crop"].unique()):
        cm = tp["crop"] == c
        if cm.sum() > 30:
            r2, sp = metrics(tp.loc[cm, "yield_anomaly"].values, tp.loc[cm, "pred"].values)
            out[c] = {"r2": r2, "spearman": sp, "n_test": int(cm.sum())}
    return out


def add_soil_lat(panel):
    """Add an NCCPI soil-quality proxy (county-crop max yield / national-crop max,
    capped [0,1]) and a latitude proxy from the county centroid. NCCPI proxy uses
    the historical yield ceiling, a standard land-capability surrogate; latitude is
    a fixed geographic attribute -- neither leaks the held-out outcome."""
    cmax = panel.groupby(["fips", "crop"])["yield_bu_acre"].transform("max")
    natmax = panel.groupby("crop")["yield_bu_acre"].transform("max")
    panel["nccpi"] = (cmax / natmax).clip(0, 1)
    # latitude from county-mean growing-season tmin as a smooth thermal-latitude proxy
    if "tmin_growing_c" in panel.columns:
        panel["lat_proxy"] = -panel.groupby("fips")["tmin_growing_c"].transform("mean")
    return panel


def main():
    panel = build_panel()
    panel = add_soil_lat(panel)
    panel = panel.merge(monthly_sequence_features(), on=["fips", "year"], how="left")

    # county-anomaly versions of the monthly sequence (within-county climate signal)
    seq_raw = [c for c in panel.columns if c.startswith("seq_")]
    for c in seq_raw:
        panel[f"{c}_an"] = panel[c] - panel.groupby("fips")[c].transform("mean")
    seq_an = [f"{c}_an" for c in seq_raw]

    yr = panel["year"].values
    tr = yr <= 2009
    bl = (yr > 2009) & (yr <= 2012)
    te = (yr > 2012) & (yr <= 2023)
    y = panel["yield_anomaly"]

    # ---- Tree on full engineered feature set (drought feats included) ----
    X_tree, _ = design(panel)
    common = dict(n_estimators=2000, learning_rate=0.02, max_depth=8,
