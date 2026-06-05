"""Revision v4: push anomaly R^2 with additional process-relevant features.

Builds on v3 (VPD, EDD30, heat-days, soil-moisture stress) by adding the thermal
and critical-period precipitation features standard in the modern crop-yield
literature but absent from the original 50-feature set:
  kdd34_growing   : killing degree-days above 34 C (lethal heat, distinct from EDD>30)
  precip_jul_anom : July precipitation anomaly (grain-fill / silking water stress)
  precip_aug_anom : August precipitation anomaly
  dtr_growing_anom: diurnal temperature range anomaly (cloud/heat-stress signal)
  vpd_aug         : August vapour pressure deficit
Trains LightGBM only (the v3 ensemble blend collapsed to LightGBM, w=[1,0,0]),
so this is the operative model. Same split (train<=2012, test 2013-2023). Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_model_v3_features import build_modern_features, add_county_anomalies, es_kpa, GROW_MONTHS

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "revision"
SEED = 42


def extra_features():
    m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet")
    m["fips"] = m["fips"].astype(str).str.zfill(5)
    for mm in GROW_MONTHS:
        m[f"tmaxc_{mm}"] = (m[f"tmax_m{mm}"] - 32) * 5 / 9
        m[f"tminc_{mm}"] = (m[f"tmin_m{mm}"] - 32) * 5 / 9
    # KDD>34C (sinusoid quadrature)
    thr = 34.0; phase = np.linspace(0, np.pi, 24); kdd = np.zeros(len(m)); dtr = np.zeros(len(m))
    for mm in GROW_MONTHS:
        tmn = m[f"tminc_{mm}"].values; tmx = m[f"tmaxc_{mm}"].values
        mid = (tmx + tmn) / 2; amp = (tmx - tmn) / 2
        dd = np.zeros_like(mid)
        for p in phase:
            dd += np.maximum(mid + amp * np.sin(p - np.pi / 2) - thr, 0)
        kdd += (dd / len(phase)) * 30.0
        dtr += (tmx - tmn)
    m["kdd34_growing"] = kdd
    m["dtr_growing"] = dtr / len(GROW_MONTHS)
    m["precip_jul"] = m["precip_m07"]; m["precip_aug"] = m["precip_m08"]
    m["vpd_aug"] = (es_kpa(m["tmaxc_08"]) - es_kpa(m["tminc_08"])).clip(lower=0)
    return m[["fips", "year", "kdd34_growing", "dtr_growing", "precip_jul",
              "precip_aug", "vpd_aug"]].copy()


def main():
    panel = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet")
    panel["fips"] = panel["fips"].astype(str).str.zfill(5)
    panel = add_county_anomalies(panel, build_modern_features())
    ex = extra_features()
    panel = panel.merge(ex, on=["fips", "year"], how="left")
    for c in ["kdd34_growing", "dtr_growing", "precip_jul", "precip_aug", "vpd_aug"]:
        panel[f"{c}_anom"] = panel[c] - panel.groupby("fips")[c].transform("mean")

    exclude = {"fips", "year", "crop", "yield_bu_acre", "yield_anomaly",
               "acres_harvested", "production"}
    fcols = [c for c in panel.columns if c not in exclude
             and panel[c].dtype.kind in "fi" and not panel[c].isna().all()]
    X = panel[fcols].fillna(0)
    X = pd.concat([X, pd.get_dummies(panel["crop"], prefix="crop")], axis=1)
    y = panel["yield_anomaly"]; yr = panel["year"].values
    tr, te = yr <= 2012, (yr > 2012) & (yr <= 2023)

    model = lgb.LGBMRegressor(objective="regression", n_estimators=2000, learning_rate=0.02,
                              max_depth=8, num_leaves=127, min_child_samples=20,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05,
                              reg_lambda=0.5, random_state=SEED, verbose=-1)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    yt = y[te].values
    r2 = 1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2)
    sp = stats.spearmanr(yt, pred).correlation
    print(f"v4 LightGBM (n_features={X.shape[1]}): R2={r2:.4f} Spearman={sp:.4f}")

    tp = panel[te].reset_index(drop=True); tp["pred"] = pred
    per = {}
    for c in sorted(tp["crop"].unique()):
        cm = tp["crop"] == c
        if cm.sum() > 30:
            o = tp.loc[cm, "yield_anomaly"].values; p = tp.loc[cm, "pred"].values
            per[c] = {"r2": float(1 - np.sum((o - p)**2)/np.sum((o-o.mean())**2)),
                      "spearman": float(stats.spearmanr(o, p).correlation)}
    out = {"r2": float(r2), "spearman": float(sp), "n_features": int(X.shape[1]),
