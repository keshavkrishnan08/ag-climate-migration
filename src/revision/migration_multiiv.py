"""Migration: overidentified multi-instrument 2SLS to tighten precision (#2).

Instead of one combined shift-share shock, use crop-specific leave-one-out
shift-share instruments (corn, soybeans, wheat) as a vector. Overidentification
improves efficiency (tighter CI), and the Hansen J statistic tests instrument
validity. Estimated on high farm-intensity counties. Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from migration_iv_bartik import build_panel, demean2
DATA_PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "revision"
PRICE = {"corn": 5.04, "soybeans": 12.29, "wheat_winter": 6.72, "wheat_spring": 7.38,
         "cotton": 0.93, "sorghum": 4.80, "barley": 5.64, "oats": 3.35}
INSTR_CROPS = ["corn", "soybeans", "wheat_winter", "sorghum"]


def crop_instruments():
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet",
                         columns=["fips", "year", "crop", "yield_anomaly", "acres_harvested"])
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    bmix = fm[fm["year"].between(2000, 2009)].groupby(["fips", "crop"])["acres_harvested"].mean().reset_index()
    tot = bmix.groupby("fips")["acres_harvested"].transform("sum")
    bmix["share"] = bmix["acres_harvested"] / tot.replace(0, np.nan)
    natl = fm.groupby(["crop", "year"])["yield_anomaly"].agg(["sum", "count"]).reset_index()
    fm2 = fm.merge(natl, on=["crop", "year"], how="left")
    fm2["g_loo"] = (fm2["sum"] - fm2["yield_anomaly"]) / (fm2["count"] - 1).clip(lower=1)
    fm2 = fm2.merge(bmix[["fips", "crop", "share"]], on=["fips", "crop"], how="left")
    fm2["term"] = fm2["share"].fillna(0) * fm2["crop"].map(PRICE).fillna(5.0) * fm2["g_loo"]
    out = None
    for c in INSTR_CROPS:
        z = (fm2[fm2["crop"] == c].groupby(["fips", "year"])["term"].sum()
             .rename(f"z_{c}").reset_index())
        out = z if out is None else out.merge(z, on=["fips", "year"], how="outer")
    return out.fillna(0)


def tsls_multi(df, y, d, zcols, ctrls):
    cols = [y, d] + zcols + ctrls
    dd = df.dropna(subset=cols).copy()
    dd = dd[np.all(np.isfinite(dd[cols].values), axis=1)]
    dm = demean2(dd, cols)
    Y = dm[y + "_dm"].values.reshape(-1, 1)
    D = dm[d + "_dm"].values.reshape(-1, 1)
    Z = np.column_stack([dm[z + "_dm"].values for z in zcols]
                        + [dm[c + "_dm"].values for c in ctrls])
    C = np.column_stack([dm[c + "_dm"].values for c in ctrls]) if ctrls else np.empty((len(dd), 0))
    Xend = np.column_stack([D, C]) if C.size else D
    # first stage
    b1, *_ = np.linalg.lstsq(Z, D, rcond=None); Dhat = Z @ b1
    ss = np.sum((Dhat - Dhat.mean()) ** 2); k = Z.shape[1]
    F = (ss / len(zcols)) / (np.sum((D - Dhat) ** 2) / (len(D) - k))
    # 2SLS
    Xhat = np.column_stack([Dhat, C]) if C.size else Dhat
    b2, *_ = np.linalg.lstsq(Xhat, Y, rcond=None)
    beta = float(b2[0, 0])
    u = (Y - Xend @ b2).ravel()
    bread = np.linalg.inv(Xhat.T @ Xhat)
    meat = np.zeros((Xhat.shape[1], Xhat.shape[1]))
    for _, idx in dd.groupby("fips").indices.items():
        Xg = Xhat[idx]; ug = u[idx]; meat += Xg.T @ np.outer(ug, ug) @ Xg
    cov = bread @ meat @ bread; se = float(np.sqrt(cov[0, 0]))
    p = 2 * (1 - stats.norm.cdf(abs(beta / se)))
    # Hansen J overid: regress 2SLS residual on instruments, n*R^2 ~ chi2(L-1)
    rr = u - u.mean()
    bz, *_ = np.linalg.lstsq(Z, rr, rcond=None); fit = Z @ bz
    R2 = np.sum((fit - fit.mean()) ** 2) / np.sum((rr - rr.mean()) ** 2)
    J = len(dd) * R2; dofJ = len(zcols) - 1
    J_p = 1 - stats.chi2.cdf(J, dofJ) if dofJ > 0 else None
    return {"beta": beta, "se": se, "p": float(p), "first_stage_F": float(F),
            "n": int(len(dd)), "n_instruments": len(zcols),
            "hansen_J": float(J), "hansen_p": float(J_p) if J_p is not None else None,
            "ci95": [beta - 1.96 * se, beta + 1.96 * se]}


