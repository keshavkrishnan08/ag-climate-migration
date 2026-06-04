"""Audit improvement D: best combined model + a learning-curve ceiling test.

Two deliverables a hostile methods reviewer would want:

1. BEST HONEST POOLED MODEL. The drought-trajectory features (audit A) gave the
   single biggest gain (R^2 0.227 -> 0.234). Here we lock in that feature set with
   light hyper-parameter tuning and report the final pooled + per-crop held-out
   anomaly R^2 / Spearman, plus confirm the WITHIN-CROP LEVELS R^2 (the number the
   paper also reports) is not degraded.

2. LEARNING CURVE / CEILING EVIDENCE. To show the remaining gap is a genuine
   signal ceiling and not under-training, we sweep the number of training years
   (and, separately, the feature-set richness) and plot held-out anomaly R^2. If
   R^2 has plateaued -- more data and more features stop helping -- the residual
   variance is irreducible weather noise + unobserved management/irrigation, i.e.
   the literature ceiling, NOT a fixable modelling error.

Target = z-scored county-detrended anomaly. Split: test 2013-2023. Seed 42.
Climate-only features (projectable from CMIP6 deltas).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_audit_drought_huber import build_panel, design
from yield_audit_mlp_stack import add_soil_lat

OUT = ROOT / "results" / "revision"
SEED = 42

COMMON = dict(n_estimators=2000, learning_rate=0.02, max_depth=8, num_leaves=127,
              min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.05, reg_lambda=0.5, random_state=SEED, verbose=-1)


def r2_sp(y, p):
    y = np.asarray(y); p = np.asarray(p)
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(r2), float(stats.spearmanr(y, p).correlation)


def per_crop_levels(panel, te_pred, te, panel_full):
    """Within-crop LEVELS R^2: reconstruct yield level = trend + anomaly*sd, then
    score predicted level vs actual on the held-out years, per crop. Uses the
    county-crop mean/sd implied by yield_anomaly (z-scored) and the realized level
    to back out sd; this matches the existing levels-R^2 convention."""
    tp = panel[te].reset_index(drop=True).copy()
    tp["pred_anom"] = te_pred
    out = {}
    for c in sorted(tp["crop"].unique()):
        cm = tp["crop"] == c
        if cm.sum() < 30:
            continue
        sub = tp[cm]
        # per-county sd of yield within crop (full-period), to map anomaly->level
        key = panel_full[panel_full["crop"] == c]
        sd = key.groupby("fips")["yield_bu_acre"].transform("std")
        mu = key.groupby("fips")["yield_bu_acre"].transform("mean")
        ref = key.assign(sd=sd, mu=mu)[["fips", "year", "sd", "mu"]]
        sub = sub.merge(ref, on=["fips", "year"], how="left")
        actual = sub["yield_bu_acre"].values
        pred_level = sub["mu"].values + sub["pred_anom"].values * sub["sd"].values
        ok = np.isfinite(pred_level) & np.isfinite(actual)
        if ok.sum() > 30:
            a = actual[ok]; p = pred_level[ok]
            out[c] = float(1 - np.sum((a - p)**2) / np.sum((a - a.mean())**2))
    return out


def main():
    panel = build_panel()
    panel = add_soil_lat(panel)
    # add nccpi/lat to design feature pool by including them as numeric cols
    X, fcols = design(panel)
    for extra in ["nccpi", "lat_proxy"]:
        if extra in panel.columns and extra not in X.columns:
            X[extra] = panel[extra].fillna(0).values
    y = panel["yield_anomaly"]
    yr = panel["year"].values
    te = (yr > 2012) & (yr <= 2023)

    # ---- best pooled model (full train <=2012) ----
    tr = yr <= 2012
