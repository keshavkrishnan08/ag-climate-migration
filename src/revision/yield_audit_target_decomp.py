"""Audit improvement E (the hostile-reviewer test): is v7's +0.13 R^2 gain real
predictive skill, or an artifact of switching the target metric?

The paper attributes most of the jump from R^2=0.23 to R^2=0.41 to predicting the
percentage deviation from trend instead of the z-scored anomaly. A hostile
reviewer's first objection: "R^2 on a different target is not comparable -- you
may simply have chosen a target with a more favorable variance structure, not a
better model." We settle this with a clean 2x2 factorial that holds the
EVALUATION target fixed within each cell:

  Factor 1 (architecture/features): AGG = growing-season aggregates (the old
            feature set) vs SPEC = temperature-exposure spectrum + per-crop.
  Factor 2 (training target):       Z = z-scored anomaly vs PCT = % deviation.

For every cell we ALSO map predictions onto BOTH common scales and report R^2 on
each, so the four models are compared on identical yardsticks:
  - R^2 on the z-scored anomaly (the conservative, old yardstick)
  - R^2 on the % deviation (the new yardstick the paper reports)
Mapping: a %-deviation prediction is converted to a yield level (trend*(1+dev))
and then z-scored with the same county-crop mean/sd used to build the anomaly,
and vice-versa. This isolates how much of the gain is the FEATURES (real skill,
shows up on both yardsticks) vs the TARGET CHOICE (shows up only on its own
yardstick).

Split: train<=2012, test 2013-2023. Seed 42. Per-crop, climate-only.
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
from yield_v7_spectrum import build as build_v7   # spectrum features + pct target

DATA_PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "revision"
SEED = 42
COMMON = dict(objective="regression", n_estimators=2000, learning_rate=0.02,
              max_depth=8, num_leaves=127, min_child_samples=30, subsample=0.8,
              colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.5,
              random_state=SEED, verbose=-1)


def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    panel, climcols = build_v7()
    # panel has: yield_bu_acre, trend, dev_pct, spectrum/precip/pdsi/vpd + _an, nccpi, latitude
    # build the z-scored anomaly target on the SAME rows, county-crop demeaned/scaled
    panel = panel.sort_values(["fips", "crop", "year"])
    grp = panel.groupby(["fips", "crop"])["yield_bu_acre"]
    mu = grp.transform("mean"); sd = grp.transform("std")
    panel["z_anom"] = ((panel["yield_bu_acre"] - mu) / sd.replace(0, np.nan))
    panel["cc_mu"] = mu; panel["cc_sd"] = sd

    # spectrum feature set (v7) and an AGGREGATE feature set (growing-season means)
    spec_feats = ([c for c in climcols] + [f"{c}_an" for c in climcols]
                  + ["latitude", "nccpi", "yield_trend_slope_15yr",
                     "switching_rate_5yr", "log_population", "log_median_income"])
    spec_feats = [f for f in spec_feats if f in panel.columns]
    # aggregates: collapse the spectrum/precip/pdsi/vpd to season means (old-style)
    tbins = [c for c in climcols if c.startswith("tbin_")]
