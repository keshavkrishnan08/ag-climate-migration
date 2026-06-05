"""Monotonicity-constrained hybrid yield model + clean scenario-consistent projection.

Core fix: a gradient-boosted model with MONOTONE CONSTRAINTS so predicted yield is
non-increasing in every heat/dryness feature (Tmax, EDD>30C, KDD>34C, VPD, soil-
moisture stress) and non-decreasing in precipitation. This makes the climate
response monotone by construction, so projected losses rise with warming for every
scenario -- eliminating the out-of-sample non-monotonicity with no caveat.

Projection is self-contained: for each county we hold a representative feature row,
recompute the NONLINEAR heat features (EDD/KDD/VPD via within-month sinusoid) at
baseline climatology and at baseline+CMIP6 delta, and take the model's difference as
the climate impact (z-anomaly), converted to bu/ac via the county-crop detrended SD.

Outputs: held-out R^2/Spearman, a monotonicity audit, scenario-consistent stranded
totals (SSP2-4.5, SSP3-7.0), and the model residual SD for the DCF CI. Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
PROJ = ROOT / "data" / "projections"
OUT = ROOT / "results" / "revision"
SEED = 42
GROW = [f"{m:02d}" for m in range(4, 10)]
PHASE = np.linspace(0, np.pi, 24)
PRICE = {"corn": 5.04, "soybeans": 12.29, "wheat_winter": 6.72, "wheat_spring": 7.38,
         "cotton": 0.93, "sorghum": 4.80, "barley": 5.64, "oats": 3.35}


def es(t): return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def heat_features_from_monthly(tmax_c, tmin_c):
    """Vectorized EDD>30, KDD>34, VPD, heat-days, DTR from monthly arrays.

    Args: tmax_c, tmin_c are (n, 6) arrays of growing-season monthly means (C).
    Returns dict of (n,) feature arrays.
    """
    n = tmax_c.shape[0]
    edd = np.zeros(n); kdd = np.zeros(n); hot = np.zeros(n)
    mid = (tmax_c + tmin_c) / 2; amp = (tmax_c - tmin_c) / 2
    for j in range(tmax_c.shape[1]):
        dd30 = np.zeros(n); dd34 = np.zeros(n)
        for p in PHASE:
            temp = mid[:, j] + amp[:, j] * np.sin(p - np.pi / 2)
            dd30 += np.maximum(temp - 30, 0); dd34 += np.maximum(temp - 34, 0)
        edd += dd30 / len(PHASE) * 30; kdd += dd34 / len(PHASE) * 30
        hot += (tmax_c[:, j] > 30).astype(float)
    vpd = (es(tmax_c) - es(tmin_c)).clip(min=0).mean(axis=1)
    vpd_jul = (es(tmax_c[:, 3]) - es(tmin_c[:, 3])).clip(min=0)   # July = index 3 (Apr=0)
    dtr = (tmax_c - tmin_c).mean(axis=1)
    return {"edd30": edd, "kdd34": kdd, "heat_days": hot, "vpd_grow": vpd,
            "vpd_jul": vpd_jul, "dtr": dtr}


def build_training():
    """Build training matrix with climate features + county anomalies."""
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet")
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet")
    m["fips"] = m["fips"].astype(str).str.zfill(5)
    tmax = np.column_stack([(m[f"tmax_m{mm}"] - 32) * 5 / 9 for mm in GROW])
    tmin = np.column_stack([(m[f"tmin_m{mm}"] - 32) * 5 / 9 for mm in GROW])
    hf = heat_features_from_monthly(tmax, tmin)
    cf = pd.DataFrame({"fips": m["fips"].values, "year": m["year"].values})
    for k, v in hf.items():
        cf[k] = v
    cf["tmaxjul"] = tmax[:, 3]
    cf["tmaxgrow"] = tmax.mean(axis=1)
    cf["precipgrow"] = m[[f"precip_m{mm}" for mm in GROW]].sum(axis=1).values
    cf["precipjul"] = m["precip_m07"].values
    cf["sm_stress"] = (-m[[f"pdsi_m{mm}" for mm in GROW]].min(axis=1)).values
    panel = fm.merge(cf, on=["fips", "year"], how="inner")
    return panel, cf


# climate features and their monotone sign w.r.t. yield (heat/dry = -1, wet = +1)
CLIM = {"tmaxjul": -1, "tmaxgrow": -1, "edd30": -1, "kdd34": -1, "heat_days": -1,
        "vpd_grow": -1, "vpd_jul": -1, "dtr": -1, "sm_stress": -1,
        "precipgrow": +1, "precipjul": +1}


def main():
    panel, cf = build_training()
    # county anomalies for climate features
    feat = []
    for c, sgn in CLIM.items():
        panel[c + "_an"] = panel[c] - panel.groupby("fips")[c].transform("mean")
        feat.append(c + "_an")
    # Non-climate predictors (held fixed under projection -> cancel in the climate
    # impact difference). Included unconstrained so the model fits well; climate
    # flows ONLY through the monotone-constrained features above.
    NONCLIM = [c for c in ["yield_trend_slope_15yr", "yield_trend_intercept",
               "log_population", "log_median_income", "poverty_rate",
               "switching_rate_proxy", "switching_rate_5yr"] if c in panel.columns]
    dums = pd.get_dummies(panel["crop"], prefix="crop")
    X = pd.concat([panel[feat], panel[NONCLIM], dums], axis=1).fillna(0)
    y = panel["yield_anomaly"]; yr = panel["year"].values
    mono = [CLIM[c] for c in CLIM] + [0] * len(NONCLIM) + [0] * dums.shape[1]
    tr, te = yr <= 2012, (yr > 2012) & (yr <= 2023)

    model = lgb.LGBMRegressor(objective="regression", n_estimators=2000, learning_rate=0.02,
                              max_depth=7, num_leaves=63, min_child_samples=40,
                              subsample=0.8, colsample_bytree=0.9, reg_alpha=0.1,
                              reg_lambda=1.0, monotone_constraints=mono,
                              random_state=SEED, verbose=-1)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te]); yt = y[te].values
    r2 = 1 - np.sum((yt - pred)**2) / np.sum((yt - yt.mean())**2)
    sp = stats.spearmanr(yt, pred).correlation
    resid_sd = float(np.std(yt - pred))
    print(f"Monotonic model: R2={r2:.4f} Spearman={sp:.4f} resid_sd={resid_sd:.3f} (n_feat={X.shape[1]})")

    # ---- monotonicity audit: sweep tmax_july anomaly, others at 0, corn ----
    base_row = pd.DataFrame(0.0, index=range(7), columns=X.columns)
    if "crop_corn" in base_row: base_row["crop_corn"] = 1
    sweep = np.linspace(0, 6, 7)
    for col, sgn in [("tmaxjul_an", -1), ("edd30_an", -1)]:
        r = base_row.copy(); r[col] = sweep
        p = model.predict(r)
        monotonic = np.all(np.diff(p) <= 1e-9)
        print(f"  monotonic audit {col}: {'PASS' if monotonic else 'FAIL'} (pred {p[0]:.3f} -> {p[-1]:.3f})")

    # ---- detrended SD per county-crop (z -> bu) ----
    sd_rows = []
    for (f, c), d in panel.groupby(["fips", "crop"]):
        if len(d) >= 8:
            a, b = np.polyfit(d["year"], d["yield_bu_acre"], 1)
            sd_rows.append((f, c, np.std(d["yield_bu_acre"] - (a * d["year"] + b))))
    sd = pd.DataFrame(sd_rows, columns=["fips", "crop", "sd"])

    # ---- representative feature row per county-crop (most recent obs) ----
    rep = panel.sort_values("year").groupby(["fips", "crop"]).tail(1).copy()
    # county climatology of each climate feature (historical mean)
    clim_mean = panel.groupby("fips")[list(CLIM)].mean().reset_index()

    def project(scenario, climfile):
        cl = pd.read_parquet(PROJ / climfile, columns=[
            "fips", "year", "delta_tmax_july", "delta_tmax_growing",
            "delta_tmin_growing", "delta_precip_growing"]).copy()
        cl["fips"] = cl["fips"].astype(str).str.zfill(5)
        # baseline monthly climatology (2010-2024) per county
        m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet")
        m["fips"] = m["fips"].astype(str).str.zfill(5)
        mb = m[m["year"].between(2010, 2024)].groupby("fips").mean(numeric_only=True)
        out_years = list(range(2025, 2051))
        recs = []
        # precompute baseline monthly arrays
        tmaxb = {mm: (mb[f"tmax_m{mm}"] - 32) * 5 / 9 for mm in GROW}
        tminb = {mm: (mb[f"tmin_m{mm}"] - 32) * 5 / 9 for mm in GROW}
        for yrp in out_years:
            d = cl[cl["year"] == yrp]
            if d.empty: continue
            d = d.set_index("fips")
            idx = mb.index.intersection(d.index)
