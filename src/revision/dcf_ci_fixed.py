"""Corrected DCF confidence interval (Reviewer 2 #1: the [$58,63B] CI was too tight).

The honest interval must carry three error sources, not just idiosyncratic county
error. We aggregate to county present-value FIRST (fixing the stranded set, so no
per-draw rectification bias), then propagate multiplicatively:
  * idiosyncratic model error  -> independent lognormal per county (cancels in sum)
  * spatially correlated error -> one common lognormal shock per state per draw
  * GCM ensemble spread        -> per-county relative spread from p10-p90
Reported as nested CIs so the widening is transparent. Seed 42.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "revision"
np.random.seed(42)
PRICE = {"corn": 5.04, "soybeans": 12.29, "wheat_winter": 6.72, "wheat_spring": 7.38,
         "cotton": 0.93, "sorghum": 4.80, "barley": 5.64, "oats": 3.35}


def main(n=2000):
    yp = pd.read_parquet(ROOT / "data" / "projections" / "yield_projections_SSP245.parquet")
    yp["fips"] = yp["fips"].astype(str).str.zfill(5)
    yp["price"] = yp["crop"].map(PRICE).fillna(4.0)
    r, H, y0 = 0.04, 30, yp["year"].min()
    yp = yp[yp["year"] <= y0 + H - 1].copy()
    yp["disc"] = 1.0 / (1 + r) ** (yp["year"] - y0 + 1)
    yp["pv_loss"] = -(yp["climate_impact_bu"] * yp["price"] * yp["acres_harvested"] * yp["disc"])
    # GCM relative spread per row -> county
    yp["gcm_rel"] = ((yp["yield_p90"] - yp["yield_p10"]) / 2.563).abs() / \
                    yp["climate_impact_bu"].abs().clip(lower=1)
    cty = yp.groupby("fips").agg(pv=("pv_loss", "sum"),
                                 gcm_rel=("gcm_rel", "mean")).reset_index()
    cty["state"] = cty["fips"].str[:2]
    stranded = cty[cty["pv"] > 0].copy()           # fixed stranded set
    pv = stranded["pv"].values
    point = pv.sum() / 1e9
    states = stranded["state"].values
    uniq = np.unique(states)
    sidx = {s: np.where(states == s)[0] for s in uniq}
    gcm_rel = np.clip(stranded["gcm_rel"].fillna(0.3).values, 0, 1.5)

    SIG_IDIO = 0.70      # ~ relative model error on the county impact (R2~0.22 on anomalies)
    SIG_SPAT = 0.30      # regional common component

    def draw(idio, spat, gcm):
        out = np.empty(n)
        for k in range(n):
            f = np.ones(len(pv))
            if idio:
                f *= np.exp(np.random.normal(0, SIG_IDIO, len(pv)) - SIG_IDIO**2 / 2)
            if spat:
                for s in uniq:
                    sh = np.exp(np.random.normal(0, SIG_SPAT) - SIG_SPAT**2 / 2)
                    f[sidx[s]] *= sh
            if gcm:
                f *= np.exp(np.random.normal(0, gcm_rel) - gcm_rel**2 / 2)
            out[k] = (pv * f).sum() / 1e9
        return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

    res = {"point_estimate_B": float(point),
           "ci_idiosyncratic_only": draw(True, False, False),
           "ci_plus_spatial": draw(True, True, False),
           "ci_full": draw(True, True, True),
           "sig_idio": SIG_IDIO, "sig_spatial": SIG_SPAT, "n_draws": n,
           "n_stranded_counties": int(len(pv))}
    print("point  $%.1fB" % point)
    print("idio   ", [round(x, 1) for x in res["ci_idiosyncratic_only"]])
