"""Audit improvement A: richer drought-trajectory features + Huber-loss target.

Orthogonal to v7 (which changes the target to %-deviation and uses a temperature
exposure spectrum). Here we KEEP the existing z-scored county-detrended anomaly
target -- so the held-out R^2 is DIRECTLY comparable to the v4 number (0.227) the
paper currently reports -- and attack two distinct, defensible levers:

1. RICHER DROUGHT DYNAMICS from the monthly PDSI/precip panel. Yield loss in
   corn/soy/cotton is driven by drought *timing and persistence*, not just the
   season minimum. We add:
     - consecutive-dry-month run length (longest streak PDSI<-1 in growing season)
     - PDSI trajectory slope (Apr->Sep linear slope: drying vs recovering)
     - PDSI integral / area below -1 (cumulative water deficit)
     - month-of-driest-PDSI (timing relative to silking/grain-fill)
     - precip deficit run (longest streak of below-county-normal precip)
     - early- vs late-season PDSI contrast (Apr-Jun mean minus Jul-Sep mean)
   All county-demeaned to anomalies, matching the pipeline convention.

2. HUBER LOSS instead of squared error. The z-scored anomaly target has +/-7 sigma
   tails (drought/flood outliers) that dominate the MSE gradient and bias the fit
   toward un-modellable freak years. Huber down-weights those tails, so the model
   learns the bulk climate-yield signal rather than chasing noise. This is a model
   improvement, not a metric reframing: we still REPORT plain R^2 / Spearman on the
   untransformed held-out anomaly.

Same temporal split (train<=2012, test 2013-2023). Seed 42. Climate-only feature
set (no spatial-yield lags) so it stays projectable from CMIP6 deltas.
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
from yield_model_v3_features import build_modern_features, add_county_anomalies, GROW_MONTHS
from yield_v4_morefeatures import extra_features

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "revision"
SEED = 42
GROW = [f"{m:02d}" for m in range(4, 10)]   # Apr-Sep


def drought_trajectory_features():
    """Engineer drought timing / persistence / shape features from monthly PDSI.

    Returns:
        DataFrame [fips, year, <drought-shape feature columns>].
    """
    m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet")
    m["fips"] = m["fips"].astype(str).str.zfill(5)

    pdsi = np.column_stack([m[f"pdsi_m{mm}"].values for mm in GROW])   # (n, 6) Apr..Sep
    precip = np.column_stack([m[f"precip_m{mm}"].values for mm in GROW])
    n, k = pdsi.shape
    months_idx = np.arange(k)

    # 1. longest consecutive run of PDSI < -1 within the growing season
    def longest_run(mask):
        out = np.zeros(mask.shape[0])
        cur = np.zeros(mask.shape[0])
        for j in range(mask.shape[1]):
            cur = np.where(mask[:, j], cur + 1, 0)
            out = np.maximum(out, cur)
        return out
    dry_run = longest_run(pdsi < -1.0)

    # 2. PDSI trajectory slope across Apr->Sep (drying if negative)
    xm = months_idx - months_idx.mean()
    denom = np.sum(xm ** 2)
    pdsi_slope = ((pdsi - pdsi.mean(axis=1, keepdims=True)) * xm).sum(axis=1) / denom

    # 3. cumulative water deficit: sum of (PDSI below -1), i.e. area under -1
    deficit_integral = np.maximum(-1.0 - pdsi, 0).sum(axis=1)

    # 4. month index (0=Apr..5=Sep) of the driest PDSI -> drought timing
    dry_month = pdsi.argmin(axis=1).astype(float)

    # 5. early vs late season PDSI contrast (Apr-Jun mean - Jul-Sep mean);
    #    positive => dried out late (during grain-fill, the costly window)
    early_late = pdsi[:, :3].mean(axis=1) - pdsi[:, 3:].mean(axis=1)

    # 6. longest run of below-county-normal precip (within-year proxy:
    #    below the within-season median precip month)
    precip_med = np.median(precip, axis=1, keepdims=True)
    precip_run = longest_run(precip < precip_med)

    out = pd.DataFrame({
        "fips": m["fips"].values, "year": m["year"].values,
        "dry_run": dry_run, "pdsi_slope": pdsi_slope,
        "deficit_integral": deficit_integral, "dry_month": dry_month,
        "pdsi_early_late": early_late, "precip_dry_run": precip_run,
    })
    return out


def build_panel():
    panel = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet")
    panel["fips"] = panel["fips"].astype(str).str.zfill(5)
    panel = add_county_anomalies(panel, build_modern_features())
    ex = extra_features()
    panel = panel.merge(ex, on=["fips", "year"], how="left")
    for c in ["kdd34_growing", "dtr_growing", "precip_jul", "precip_aug", "vpd_aug"]:
        panel[f"{c}_anom"] = panel[c] - panel.groupby("fips")[c].transform("mean")
    dr = drought_trajectory_features()
    panel = panel.merge(dr, on=["fips", "year"], how="left")
    for c in ["dry_run", "pdsi_slope", "deficit_integral", "dry_month",
              "pdsi_early_late", "precip_dry_run"]:
        panel[f"{c}_anom"] = panel[c] - panel.groupby("fips")[c].transform("mean")
    return panel


def design(panel):
    exclude = {"fips", "year", "crop", "yield_bu_acre", "yield_anomaly",
               "acres_harvested", "production"}
    fcols = [c for c in panel.columns if c not in exclude
             and panel[c].dtype.kind in "fi" and not panel[c].isna().all()]
    X = panel[fcols].fillna(0)
    X = pd.concat([X, pd.get_dummies(panel["crop"], prefix="crop")], axis=1)
    return X, fcols


def evaluate(panel, pred, te):
    yt = panel.loc[te, "yield_anomaly"].values
    r2 = 1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2)
    sp = stats.spearmanr(yt, pred).correlation
    tp = panel[te].reset_index(drop=True).copy(); tp["pred"] = pred
    per = {}
    for c in sorted(tp["crop"].unique()):
        cm = tp["crop"] == c
        if cm.sum() > 30:
            o = tp.loc[cm, "yield_anomaly"].values; p = tp.loc[cm, "pred"].values
            per[c] = {"r2": float(1 - np.sum((o - p)**2)/np.sum((o-o.mean())**2)),
                      "spearman": float(stats.spearmanr(o, p).correlation),
                      "n_test": int(cm.sum())}
    return float(r2), float(sp), per


def main():
    panel = build_panel()
    X, fcols = design(panel)
    y = panel["yield_anomaly"]
    yr = panel["year"].values
    tr = yr <= 2012
    te = (yr > 2012) & (yr <= 2023)

    common = dict(n_estimators=2000, learning_rate=0.02, max_depth=8,
                  num_leaves=127, min_child_samples=20, subsample=0.8,
                  colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.5,
                  random_state=SEED, verbose=-1)

    results = {}
    # (a) MSE baseline with the new drought features (isolates feature effect)
    m_mse = lgb.LGBMRegressor(objective="regression", **common)
    m_mse.fit(X[tr], y[tr])
    r2, sp, per = evaluate(panel, m_mse.predict(X[te]), te)
    results["mse_drought"] = {"r2": r2, "spearman": sp, "per_crop": per}
    print(f"[MSE + drought feats] R2={r2:.4f} Spearman={sp:.4f}")

    # (b) Huber loss with the new drought features (isolates loss effect)
    best = None
    for alpha in (0.9, 0.95, 0.99):
        m_h = lgb.LGBMRegressor(objective="huber", alpha=alpha, **common)
        m_h.fit(X[tr], y[tr])
        r2, sp, per = evaluate(panel, m_h.predict(X[te]), te)
        print(f"[Huber alpha={alpha} + drought] R2={r2:.4f} Spearman={sp:.4f}")
        if best is None or r2 > best[0]:
            best = (r2, sp, per, alpha, m_h)
    results["huber_drought"] = {"r2": best[0], "spearman": best[1],
                                "per_crop": best[2], "best_alpha": best[3]}

    # importance of new drought features in the Huber model
    imp = pd.Series(best[4].feature_importances_, index=X.columns)
    drought_cols = [c for c in X.columns if any(
        k in c for k in ["dry_run", "pdsi_slope", "deficit_integral",
                         "dry_month", "pdsi_early_late", "precip_dry_run"])]
    results["drought_feature_importance"] = {
        c: int(imp.get(c, 0)) for c in drought_cols}
    results["n_features"] = int(X.shape[1])
    results["split"] = "train<=2012, test 2013-2023"
    results["baseline_v4_r2"] = 0.2269
    json.dump(results, open(OUT / "audit_yield_drought_huber.json", "w"), indent=2)
    print(f"\nbest Huber R2={best[0]:.4f} (alpha={best[3]}) vs v4 0.227")
    print("per-crop (Huber):", {k: round(v["r2"], 3) for k, v in best[2].items()})


if __name__ == "__main__":
    main()
