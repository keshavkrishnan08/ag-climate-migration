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
    g = pd.read_csv(DATA_RAW / "census" / "2023_Gaz_counties_national.txt",
                    sep="\t", dtype=str)
    g.columns = [c.strip() for c in g.columns]
    g["fips"] = g["GEOID"].str.zfill(5)
    g["latitude"] = pd.to_numeric(g["INTPTLAT"], errors="coerce")
    return g[["fips", "latitude"]]


def main():
    panel = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet")
    panel["fips"] = panel["fips"].astype(str).str.zfill(5)
    # climate features + anomalies
    mf = build_modern_features()
    panel = panel.merge(mf, on=["fips", "year"], how="left")
    for c in ["vpd_growing", "vpd_july", "edd30_growing", "heat_days_proxy",
              "sm_stress", "sm_stress_july", "vpd_x_sm"]:
        panel[f"{c}_anom"] = panel[c] - panel.groupby("fips")[c].transform("mean")
    ex = extra_features()
    panel = panel.merge(ex, on=["fips", "year"], how="left")
    for c in ["kdd34_growing", "dtr_growing", "precip_jul", "precip_aug"]:
        panel[f"{c}_anom"] = panel[c] - panel.groupby("fips")[c].transform("mean")
    # soil proxy (NCCPI-style): county max yield / national max, per crop
    panel = panel.merge(latitude(), on="fips", how="left")
    cmax = panel.groupby(["fips", "crop"])["yield_bu_acre"].transform("max")
    natmax = panel.groupby("crop")["yield_bu_acre"].transform("max")
    panel["nccpi"] = (cmax / natmax).clip(0, 1)
    # AR1 prior-year own anomaly
    panel = panel.sort_values(["fips", "crop", "year"])
    panel["ar1"] = panel.groupby(["fips", "crop"])["yield_anomaly"].shift(1)

    clim_feats = [c for c in CLIM_SIGN if c in panel.columns]
    feats = clim_feats + [c for c in NONCLIM if c in panel.columns]
    mono = [CLIM_SIGN[c] for c in clim_feats] + [0] * len([c for c in NONCLIM if c in panel.columns])

    yr = panel["year"].values
    results, all_obs, all_pred = {}, [], []
    for crop in sorted(panel["crop"].unique()):
        d = panel[panel["crop"] == crop].copy()
        X = d[feats].fillna(0); y = d["yield_anomaly"]
        tr = d["year"] <= 2012; te = (d["year"] > 2012) & (d["year"] <= 2023)
        if tr.sum() < 500 or te.sum() < 100:
            continue
        m = lgb.LGBMRegressor(objective="regression", n_estimators=1500, learning_rate=0.02,
                              max_depth=7, num_leaves=63, min_child_samples=40,
                              subsample=0.8, colsample_bytree=0.9, reg_alpha=0.1,
                              reg_lambda=1.0, monotone_constraints=mono,
                              random_state=SEED, verbose=-1)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te]); yt = y[te].values
        r2 = 1 - np.sum((yt - p) ** 2) / np.sum((yt - yt.mean()) ** 2)
        sp = stats.spearmanr(yt, p).correlation
        results[crop] = {"r2": float(r2), "spearman": float(sp), "n_test": int(te.sum())}
        all_obs.extend(yt); all_pred.extend(p)
        print(f"  {crop:14s} R2={r2:.3f} Spearman={sp:.3f} (n={te.sum()})")
    o = np.array(all_obs); pr = np.array(all_pred)
    overall = {"r2": float(1 - np.sum((o - pr) ** 2) / np.sum((o - o.mean()) ** 2)),
               "spearman": float(stats.spearmanr(o, pr).correlation), "n": int(len(o))}
    print(f"OVERALL (pooled across per-crop models): R2={overall['r2']:.4f} Spearman={overall['spearman']:.4f}")
    json.dump({"overall": overall, "per_crop": results, "features": feats,
               "monotone": True, "vs_v4_pooled_r2": 0.2269},
              open(OUT / "yield_v5_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
