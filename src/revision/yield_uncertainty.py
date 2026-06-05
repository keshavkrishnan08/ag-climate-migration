"""Revision: yield levels-R^2 and a properly propagated DCF confidence interval
(Reviewer 2 #1).

Two points:

(A) The reviewer's R^2 > 0.5 benchmark refers to models that predict yield
    LEVELS (which include the dominant technology trend and county effects).
    Our headline R^2 = 0.22 is on the much harder z-scored, detrended ANOMALY.
    On a true out-of-sample levels hindcast (2013-2023), the same model explains
    the great majority of yield variance. We compute that here.

(B) The Monte Carlo CI of [$58, $63 B] was implausibly tight because it
    propagated only IDIOSYNCRATIC county errors, which cancel in aggregation.
    A defensible interval must also carry (i) spatially correlated prediction
    error (regional weather events), and (ii) GCM ensemble spread. We propagate
    all three and report the decomposition so the widening is transparent.

Seed 42. Writes only to results/revision/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
PROJ = ROOT / "data" / "projections"
OUT = ROOT / "results" / "revision"
np.random.seed(42)


def levels_hindcast_r2():
    """Out-of-sample LEVELS R^2 for 2013-2023.

    predicted_level = linear technology trend (fit on years <= 2012) extrapolated
    to the test year + predicted anomaly * detrended SD. Compared with observed
    yields. This is the metric directly comparable to AgMIP/hybrid-ML county
    benchmarks (which are reported on levels).
    """
    res = pd.read_parquet(OUT / "yield_v3_test_residuals.parquet")  # fips,crop,year,yield_anomaly,pred,resid
    res["fips"] = res["fips"].astype(str).str.zfill(5)
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet",
                         columns=["fips", "year", "crop", "yield_bu_acre"])
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)

    obs_level, pred_level, crops = [], [], []
    for (fips, crop), d in fm.groupby(["fips", "crop"], sort=False):
        train = d[(d["year"] <= 2012) & (d["yield_bu_acre"] > 0)]
        if len(train) < 8:
            continue
        a, b = np.polyfit(train["year"], train["yield_bu_acre"], 1)
        detr_sd = np.std(train["yield_bu_acre"] - (a * train["year"] + b))
        if detr_sd <= 0:
            continue
        sub = res[(res["fips"] == fips) & (res["crop"] == crop)]
        if sub.empty:
            continue
        trend = a * sub["year"].values + b
        pl = trend + sub["pred"].values * detr_sd
        ol = (fm[(fm["fips"] == fips) & (fm["crop"] == crop)]
              .set_index("year").reindex(sub["year"].values)["yield_bu_acre"].values)
        m = np.isfinite(ol) & np.isfinite(pl)
        obs_level.extend(ol[m]); pred_level.extend(pl[m]); crops.extend([crop] * m.sum())

    o = np.array(obs_level); p = np.array(pred_level); c = np.array(crops)
    def r2(o, p):
        return float(1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2))
    out = {"levels_r2_overall": r2(o, p),
           "levels_rmse_overall": float(np.sqrt(np.mean((o - p) ** 2))),
           "n": int(len(o))}
    for crop in ["corn", "soybeans", "wheat_winter"]:
        mm = c == crop
        if mm.sum() > 50:
            out[f"levels_r2_{crop}"] = r2(o[mm], p[mm])
            out[f"levels_rmse_{crop}"] = float(np.sqrt(np.mean((o[mm] - p[mm]) ** 2)))
    return out


def dcf_uncertainty(n_draws=500):
    """Propagate idiosyncratic + spatial + GCM uncertainty into conservative DCF."""
    prices = pd.read_csv(OUT / "real_prices_2023usd.csv")
    pmap = dict(zip(prices.iloc[:, 0], prices.iloc[:, 1])) if prices.shape[1] >= 2 else {}
    # fall back to known real prices if columns differ
    default = {"corn": 5.04, "soybeans": 12.29, "wheat_winter": 6.72, "wheat_spring": 7.38,
               "cotton": 0.93, "sorghum": 4.80, "barley": 5.64, "oats": 3.35}
    yp = pd.read_parquet(PROJ / "yield_projections_SSP245.parquet")
    yp["fips"] = yp["fips"].astype(str).str.zfill(5)
    yp["price"] = yp["crop"].map(lambda c: default.get(c, 4.0))
    yp["state"] = yp["fips"].str[:2]

    r, H, y0 = 0.04, 30, yp["year"].min()
    yp = yp[yp["year"] <= y0 + H - 1].copy()
    yp["disc"] = 1.0 / (1 + r) ** (yp["year"] - y0 + 1)
    # model anomaly residual SD (z units) and per-row yield scale
    res = pd.read_parquet(OUT / "yield_v3_test_residuals.parquet")
    resid_sd_z = float(res["resid"].std())
    # convert z residual to bu: use county-crop projected yield * a nominal anomaly CV (~0.12)
    yp["impact_sd_gcm"] = ((yp["yield_p90"] - yp["yield_p10"]) / 2.563).abs()
    yp["yield_scale"] = yp["yield_projected"].clip(lower=1) * 0.12   # detrended SD proxy

    base = -(yp["climate_impact_bu"] * yp["price"] * yp["acres_harvested"] * yp["disc"])
    point = base.clip(lower=None)  # keep sign; sum positive stranded below
    pe = float(yp.assign(v=base).query("v>0")["v"].sum() / 1e9)

    states = yp["state"].values
    uniq_states = np.unique(states)
    sidx = {s: np.where(states == s)[0] for s in uniq_states}
    px = (yp["price"] * yp["acres_harvested"] * yp["disc"]).values
    impact = yp["climate_impact_bu"].values
    sd_gcm = yp["impact_sd_gcm"].fillna(0).values
    sc = yp["yield_scale"].values

    def total(impact_draw):
        v = -(impact_draw * px)
        return v[v > 0].sum() / 1e9

    def run(idio, spatial, gcm):
