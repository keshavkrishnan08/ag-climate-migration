"""Final yield model: temperature spectrum + monthly climate + soil + latitude +
IRRIGATION SHARE (sub-county management, from NASS practice records). Trains both
the levels model (AgMIP-comparable, target R^2>=0.5) and the %-deviation model.
Caches the engineered feature matrix so it is built only once. Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_v7_spectrum import temperature_spectrum, es, GROW
from yield_v5_percrop import latitude
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "revision"
SEED = 42
CACHE = OUT / "yield_features_full.parquet"


def build_features():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet",
                         columns=["fips", "year", "crop", "yield_bu_acre",
                                  "log_population", "log_median_income", "yield_trend_slope_15yr"])
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    fm = fm[fm["yield_bu_acre"] > 0].copy()
    m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet")
    m["fips"] = m["fips"].astype(str).str.zfill(5)
    tmax = np.column_stack([(m[f"tmax_m{mm}"] - 32) * 5 / 9 for mm in GROW])
    tmin = np.column_stack([(m[f"tmin_m{mm}"] - 32) * 5 / 9 for mm in GROW])
    spec = temperature_spectrum(tmax, tmin)
    cf = pd.DataFrame({"fips": m["fips"].values, "year": m["year"].values})
    for b in range(spec.shape[1]):
        cf[f"tbin_{b}"] = spec[:, b]
    for mm in GROW:
        cf[f"precip_{mm}"] = m[f"precip_m{mm}"].values
        cf[f"pdsi_{mm}"] = m[f"pdsi_m{mm}"].values
        tc = (m[f"tmax_m{mm}"] - 32) * 5 / 9; tn = (m[f"tmin_m{mm}"] - 32) * 5 / 9
        cf[f"vpd_{mm}"] = (es(tc) - es(tn)).clip(lower=0).values
    panel = fm.merge(cf, on=["fips", "year"], how="left").merge(latitude(), on="fips", how="left")
    cmax = panel.groupby(["fips", "crop"])["yield_bu_acre"].transform("max")
    natmax = panel.groupby("crop")["yield_bu_acre"].transform("max")
    panel["nccpi"] = (cmax / natmax).clip(0, 1)
    # irrigation propensity (time-invariant, projectable); 0 = rainfed
    irr = pd.read_parquet(OUT / "irrigation_share.parquet")[["fips", "crop", "irr_prop"]].drop_duplicates()
    panel = panel.merge(irr, on=["fips", "crop"], how="left")
    panel["irr_prop"] = panel["irr_prop"].fillna(0.0)
    panel.to_parquet(CACHE, index=False)
    return panel


def trend_pct(panel):
    tr = panel[panel["year"] <= 2012]
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
    p = panel.merge(agg[["fips", "crop", "slope", "intercept"]], on=["fips", "crop"], how="inner")
    p["trend"] = p["intercept"] + p["slope"]*p["year"]
    p = p[p["trend"] > 0].copy()
    p["dev_pct"] = (p["yield_bu_acre"]/p["trend"] - 1).clip(-1, 1)
    return p


def run(panel, target, feats, label):
    res, ao, ap = {}, [], []
    for crop in sorted(panel["crop"].unique()):
        d = panel[(panel["crop"] == crop) & panel[target].notna()]
        X = d[feats].fillna(0); y = d[target]
        tr = d["year"] <= 2012; te = (d["year"] > 2012) & (d["year"] <= 2023)
        if tr.sum() < 500 or te.sum() < 100:
            continue
        mdl = lgb.LGBMRegressor(objective="regression", n_estimators=2500, learning_rate=0.02,
                                max_depth=8, num_leaves=127, min_child_samples=30, subsample=0.8,
                                colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.5,
                                random_state=SEED, verbose=-1)
        mdl.fit(X[tr], y[tr]); p = mdl.predict(X[te]); yt = y[te].values
        r2 = 1 - np.sum((yt-p)**2)/np.sum((yt-yt.mean())**2)
        res[crop] = {"r2": float(r2), "spearman": float(stats.spearmanr(yt, p).correlation)}
        ao.extend(yt); ap.extend(p)
        print(f"  [{label}] {crop:14s} R2={r2:.3f} rho={res[crop]['spearman']:.3f}")
    o = np.array(ao); pr = np.array(ap)
    overall = {"r2": float(1-np.sum((o-pr)**2)/np.sum((o-o.mean())**2)),
               "spearman": float(stats.spearmanr(o, pr).correlation),
               "n_above_0.5": sum(1 for v in res.values() if v["r2"] >= 0.5), "n_crops": len(res),
               "median_r2": float(np.median([v["r2"] for v in res.values()]))}
    print(f"  [{label}] OVERALL R2={overall['r2']:.4f} | crops>=0.5: {overall['n_above_0.5']}/{overall['n_crops']} median {overall['median_r2']:.3f}")
    return {"overall": overall, "per_crop": res}


def main():
    panel = build_features()
    clim = [c for c in panel.columns if c.startswith(("tbin_", "precip_", "pdsi_", "vpd_"))]
    extra = ["latitude", "nccpi", "irr_prop", "log_population", "log_median_income", "yield_trend_slope_15yr"]
    # LEVELS model (add tech_time)
    panel["tech_time"] = panel["year"] - 1980
    lev = run(panel, "yield_bu_acre", clim + extra + ["tech_time"], "LEVELS+irr")
    # %-deviation model
    pdf = trend_pct(panel)
    dev = run(pdf, "dev_pct", clim + extra, "PCTDEV+irr")
    json.dump({"levels": lev, "pct_dev": dev, "irrigation_feature": True},
