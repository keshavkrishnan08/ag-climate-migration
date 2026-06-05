"""Audit improvement C2: best-honest cotton model on the %-deviation target
(matching the paper's v7 yardstick), with the irrigation diagnosis.

The paper reports cotton deviation R^2 = 0.164 (Spearman 0.483). We test whether
a cotton-DEDICATED model on the same %-deviation target, enriched with the
drought-trajectory features and a soil/irrigation proxy, beats that, and we
quantify how much of cotton's irreducible variance is irrigation.

Approach:
1. Build the v7 panel (spectrum + monthly precip/PDSI/VPD, NCCPI, latitude,
   %-deviation target) and ADD the drought-trajectory features (dry-run,
   PDSI slope, deficit integral, timing, early-late contrast).
2. Train a cotton-only LightGBM with mild regularization. Report deviation R^2 /
   Spearman vs the paper's 0.164.
3. Split cotton counties into rainfed vs irrigated by TRAIN-period yield-PDSI
   coupling and report skill separately -- the evidence that the cotton ceiling
   is an irrigation effect, not a model defect.

Split: train<=2012, test 2013-2023. Seed 42.
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
from yield_v7_spectrum import build as build_v7
from yield_audit_drought_huber import drought_trajectory_features

OUT = ROOT / "results" / "revision"
SEED = 42


def r2_sp(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(r2), float(stats.spearmanr(y, p).correlation)


def main():
    panel, climcols = build_v7()
    dr = drought_trajectory_features()
    panel = panel.merge(dr, on=["fips", "year"], how="left")
    dr_cols = ["dry_run", "pdsi_slope", "deficit_integral", "dry_month",
               "pdsi_early_late", "precip_dry_run"]
    for c in dr_cols:
        panel[f"{c}_an"] = panel[c] - panel.groupby("fips")[c].transform("mean")

    cot = panel[(panel["crop"] == "cotton") & panel["dev_pct"].notna()].copy()
    feats = ([c for c in climcols] + [f"{c}_an" for c in climcols]
             + dr_cols + [f"{c}_an" for c in dr_cols]
             + ["latitude", "nccpi", "yield_trend_slope_15yr", "switching_rate_5yr",
                "log_population", "log_median_income"])
    feats = [f for f in feats if f in cot.columns]
    X = cot[feats].fillna(0); y = cot["dev_pct"]
