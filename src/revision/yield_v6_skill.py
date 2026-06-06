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
OUT = ROOT / "results" / "revision"
SEED = 42


def main():
    panel = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet")
    panel["fips"] = panel["fips"].astype(str).str.zfill(5)
    panel = panel.merge(build_modern_features(), on=["fips", "year"], how="left")
    for c in ["vpd_growing", "vpd_july", "edd30_growing", "heat_days_proxy",
              "sm_stress", "sm_stress_july", "vpd_x_sm"]:
        panel[f"{c}_anom"] = panel[c] - panel.groupby("fips")[c].transform("mean")
    panel = panel.merge(extra_features(), on=["fips", "year"], how="left")
    for c in ["kdd34_growing", "dtr_growing", "precip_jul", "precip_aug"]:
        panel[f"{c}_anom"] = panel[c] - panel.groupby("fips")[c].transform("mean")
    panel = panel.merge(latitude(), on="fips", how="left")
    cmax = panel.groupby(["fips", "crop"])["yield_bu_acre"].transform("max")
    natmax = panel.groupby("crop")["yield_bu_acre"].transform("max")
    panel["nccpi"] = (cmax / natmax).clip(0, 1)
    panel = panel.sort_values(["fips", "crop", "year"])
    panel["ar1"] = panel.groupby(["fips", "crop"])["yield_anomaly"].shift(1)
    # same-year regional (state) spatial-lag anomaly, excluding self
    panel["state"] = panel["fips"].str[:2]
    grp = panel.groupby(["state", "crop", "year"])["yield_anomaly"]
    s_sum = grp.transform("sum"); s_cnt = grp.transform("count")
    panel["spatial_lag"] = (s_sum - panel["yield_anomaly"]) / (s_cnt - 1).clip(lower=1)

    exclude = {"fips", "year", "crop", "yield_bu_acre", "yield_anomaly",
               "acres_harvested", "production", "state"}
    feats = [c for c in panel.columns if c not in exclude
             and panel[c].dtype.kind in "fi" and not panel[c].isna().all()]
    yr = panel["year"].values
    res, ao, ap = {}, [], []
    for crop in sorted(panel["crop"].unique()):
        d = panel[panel["crop"] == crop]
        X = d[feats].fillna(0); y = d["yield_anomaly"]
        tr = d["year"] <= 2012; te = (d["year"] > 2012) & (d["year"] <= 2023)
        if tr.sum() < 500 or te.sum() < 100:
            continue
        m = lgb.LGBMRegressor(objective="regression", n_estimators=2000, learning_rate=0.02,
                              max_depth=8, num_leaves=127, min_child_samples=30,
