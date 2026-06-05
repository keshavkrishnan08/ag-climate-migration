"""Audit improvement C: a defensible cotton-only yield model + irrigation diagnosis.

Cotton anomaly R^2 sits near zero in the pooled model. The hypothesis a hostile
reviewer would demand we test: cotton's interannual yield is governed by
irrigation and soil, not by rainfed climate variability, so climate features
cannot explain it where water is supplied artificially. We test this directly.

Steps:
1. Build a cotton-ONLY model on the full engineered climate feature set
   (modern agro-climatic + drought-trajectory) plus an NCCPI soil-quality proxy
   (county yield ceiling / national ceiling) and a latitude proxy. Held-out
   2013-2023 R^2 / Spearman. This is the best honest climate-only cotton number.
2. Classify counties as IRRIGATION-DOMINATED vs RAINFED using an observable,
   non-circular signal: the historical sensitivity of county yield to growing-
   season PDSI (computed on the TRAIN period only). Irrigated counties show
   near-zero yield-PDSI coupling; rainfed counties show strong coupling.
3. Re-evaluate the model's held-out skill SEPARATELY on rainfed vs irrigated
   cotton counties. If climate skill is concentrated in rainfed counties, that
   is direct evidence the cotton ceiling is irrigation-driven, not a model defect.

Target = z-scored county-detrended anomaly (comparable to the pooled 0.23).
Split: train<=2012, test 2013-2023. Seed 42. Climate-only (projectable).
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
from yield_audit_drought_huber import build_panel
from yield_audit_mlp_stack import add_soil_lat

OUT = ROOT / "results" / "revision"
SEED = 42


def r2_sp(y, p):
    y = np.asarray(y); p = np.asarray(p)
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(r2), float(stats.spearmanr(y, p).correlation)


def main():
    panel = build_panel()
    panel = add_soil_lat(panel)
    cot = panel[panel["crop"] == "cotton"].copy()

    exclude = {"fips", "year", "crop", "yield_bu_acre", "yield_anomaly",
               "acres_harvested", "production"}
    fcols = [c for c in cot.columns if c not in exclude
             and cot[c].dtype.kind in "fi" and not cot[c].isna().all()]
    X = cot[fcols].fillna(0)
    y = cot["yield_anomaly"]
    yr = cot["year"].values
    tr = yr <= 2012
    te = (yr > 2012) & (yr <= 2023)

    model = lgb.LGBMRegressor(objective="regression", n_estimators=1500,
                              learning_rate=0.02, max_depth=6, num_leaves=63,
                              min_child_samples=30, subsample=0.8,
                              colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=SEED, verbose=-1)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    r2_all, sp_all = r2_sp(y[te].values, pred)
    print(f"[cotton-only, all counties] R2={r2_all:.4f} Spearman={sp_all:.4f} "
          f"n_test={int(te.sum())} n_feat={X.shape[1]}")

    # --- irrigation classifier: yield-PDSI coupling on TRAIN period only ---
    # rainfed cotton => yield rises with wetter PDSI => positive corr(anomaly, pdsi)
    train_cot = cot[tr]
    coupling = {}
    pdsi_col = "pdsi_growing" if "pdsi_growing" in cot.columns else None
    for fips, g in train_cot.groupby("fips"):
        if len(g) >= 8 and pdsi_col and g[pdsi_col].std() > 0:
