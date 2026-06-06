"""Fair comparison: same %-deviation target, OLD growing-season aggregate features
(no temperature spectrum). Isolates how much the spectrum representation adds vs
the original aggregates, holding the target fixed.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_v5_percrop import latitude
DATA_PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "revision"
SEED = 42


def main():
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet")
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    fm = fm[fm["yield_bu_acre"] > 0].copy()
    # same %-deviation target as v7 (closed-form trend on <=2012)
    tr = fm[fm["year"] <= 2012]
    g = tr.groupby(["fips", "crop"])
    agg = g.agg(n=("year", "size"), sx=("year", "sum"), sy=("yield_bu_acre", "sum"),
                sxx=("year", lambda s: (s.astype(float)**2).sum())).reset_index()
    sxy = (tr.assign(xy=tr["year"]*tr["yield_bu_acre"]).groupby(["fips", "crop"])["xy"]
           .sum().reset_index(name="sxy"))
    agg = agg.merge(sxy, on=["fips", "crop"])
    den = agg["n"]*agg["sxx"] - agg["sx"]**2
    agg["slope"] = np.where(den != 0, (agg["n"]*agg["sxy"] - agg["sx"]*agg["sy"])/den, np.nan)
    agg["intercept"] = (agg["sy"] - agg["slope"]*agg["sx"])/agg["n"]
    agg = agg[(agg["n"] >= 8) & agg["slope"].notna()]
    fm = fm.merge(agg[["fips", "crop", "slope", "intercept"]], on=["fips", "crop"], how="inner")
    fm["trend"] = fm["intercept"] + fm["slope"]*fm["year"]
    fm = fm[fm["trend"] > 0].copy()
    fm["dev_pct"] = (fm["yield_bu_acre"]/fm["trend"] - 1).clip(-1, 1)
    fm = fm.merge(latitude(), on="fips", how="left")
    cmax = fm.groupby(["fips", "crop"])["yield_bu_acre"].transform("max")
    natmax = fm.groupby("crop")["yield_bu_acre"].transform("max")
    fm["nccpi"] = (cmax/natmax).clip(0, 1)

    # OLD aggregate features only (growing-season means, no monthly spectrum)
    agg_feats = [c for c in ["tmax_july_c", "tmax_growing_c", "tmin_growing_c",
                 "precip_growing", "pdsi_growing", "cdd_annual", "gdd_corn", "gdd_soybeans",
                 "gdd_wheat_winter", "gdd_cotton", "gdd_sorghum", "extreme_heat_months",
                 "tmax_july_c_anomaly", "precip_growing_anomaly", "pdsi_growing_anomaly",
                 "latitude", "nccpi", "yield_trend_slope_15yr"] if c in fm.columns]
    res, ao, ap = {}, [], []
    for crop in sorted(fm["crop"].unique()):
        d = fm[(fm["crop"] == crop) & fm["dev_pct"].notna()]
        X = d[agg_feats].fillna(0); y = d["dev_pct"]
        trn = d["year"] <= 2012; te = (d["year"] > 2012) & (d["year"] <= 2023)
        if trn.sum() < 500 or te.sum() < 100:
            continue
        m = lgb.LGBMRegressor(objective="regression", n_estimators=2000, learning_rate=0.02,
                              max_depth=8, num_leaves=127, min_child_samples=30, subsample=0.8,
                              colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.5,
                              random_state=SEED, verbose=-1)
        m.fit(X[trn], y[trn]); p = m.predict(X[te]); yt = y[te].values
        r2 = 1 - np.sum((yt-p)**2)/np.sum((yt-yt.mean())**2)
        res[crop] = {"r2": float(r2), "spearman": float(stats.spearmanr(yt, p).correlation)}
        ao.extend(yt); ap.extend(p)
        print(f"  {crop:14s} R2={r2:.3f}")
    o = np.array(ao); pr = np.array(ap)
    overall = {"r2": float(1-np.sum((o-pr)**2)/np.sum((o-o.mean())**2)),
               "spearman": float(stats.spearmanr(o, pr).correlation)}
    print(f"BASELINE (aggregates, same %-dev target) R2={overall['r2']:.4f} Spearman={overall['spearman']:.4f}")
    json.dump({"overall": overall, "per_crop": res, "note": "aggregate features, same %-dev target as v7"},
              open(OUT / "yield_v7_baseline_metrics.json", "w"), indent=2)
