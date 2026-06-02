"""Formally model endogenous coverage selection (Reviewer 2, Major 2).

R2: "farmers in climate-stressed counties may rationally select higher coverage
because indemnity probability is higher, biasing the implicit-transfer estimate in
a direction the uniform calculation cannot capture."

We do two things:
(1) TEST the relationship: regress the acreage-weighted coverage election on a
    county climate-stress measure (projected yield decline), with crop fixed
    effects and acreage weights. A positive, significant coefficient confirms
    stressed counties up-select coverage.
(2) QUANTIFY the bias: recompute the rolling-APH residual mispricing and implicit
    transfer using each county-crop's ACTUAL elected coverage (endogenous) versus a
    counterfactual UNIFORM coverage (national mean). The difference is the part of
    the transfer that a uniform-coverage calculation cannot capture.

Seed 42; writes only to results/revision/.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from insurance_rolling_aph import build_rma_county_crop
from insurance_fast import build_paths, simulate_fast
PROJ = ROOT / "data" / "projections"
OUT = ROOT / "results" / "revision"
np.random.seed(42)


def climate_stress():
    """County climate-stress = acreage-weighted projected yield decline (2040-2050),
    expressed as fraction of baseline yield (positive = more stressed)."""
    yp = pd.read_parquet(PROJ / "yield_projections_SSP245.parquet",
                         columns=["fips", "year", "crop", "climate_impact_bu",
                                  "yield_baseline", "acres_harvested"])
    yp["fips"] = yp["fips"].astype(str).str.zfill(5)
    late = yp[yp["year"].between(2040, 2050)].copy()
    late["decline_frac"] = -late["climate_impact_bu"] / late["yield_baseline"].clip(lower=1)
    g = late.groupby(["fips", "crop"]).apply(
        lambda d: np.average(d["decline_frac"], weights=d["acres_harvested"].clip(lower=1e-6))
        if d["acres_harvested"].sum() > 0 else d["decline_frac"].mean(),
        include_groups=False).rename("stress").reset_index()
    return g


def wls(y, X, w):
    """Weighted least squares with HC1 SE. X includes intercept."""
    W = np.sqrt(w)
    Xw = X * W[:, None]; yw = y * W
    b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ b
    XtWX_inv = np.linalg.inv((X * w[:, None]).T @ X)
    meat = (X * (w * resid**2)[:, None]).T @ X
    cov = XtWX_inv @ meat @ XtWX_inv
    se = np.sqrt(np.diag(cov))
    return b, se


def main():
    rma = build_rma_county_crop()              # fips, crop, cov_wt, insured_acres, ...
    stress = climate_stress()
    df = rma.merge(stress, on=["fips", "crop"], how="inner").dropna(subset=["cov_wt", "stress"])
    df = df[(df["insured_acres"] > 0) & np.isfinite(df["stress"])]

    # (1) Test: cov_wt ~ stress + crop FE, acreage-weighted
    dummies = pd.get_dummies(df["crop"], prefix="c", drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(df)), df["stress"].values, dummies.values])
    b, se = wls(df["cov_wt"].values, X, df["insured_acres"].values)
    beta_stress, se_stress = b[1], se[1]
    t = beta_stress / se_stress
    p = 2 * (1 - stats.norm.cdf(abs(t)))
    # interpretable: coverage change for a +10pp projected decline
    effect_10pp = beta_stress * 0.10

    print("=== (1) Endogenous coverage selection test (WLS, crop FE, acre-weighted) ===")
    print(f"  beta(stress) = {beta_stress:+.4f}  SE={se_stress:.4f}  p={p:.4f}  n={len(df)}")
    print(f"  => a +10pp projected yield decline is associated with {effect_10pp*100:+.2f}pp coverage")

    # (2) Quantify bias: transfer under ACTUAL vs UNIFORM coverage
    paths, cv = build_paths("SSP245")
    res_actual = simulate_fast(rma, paths, cv, aph_window=10)            # endogenous elections
    natl_cov = float(np.average(rma["cov_wt"], weights=rma["insured_acres"]))
    rma_uniform = rma.copy(); rma_uniform["cov_wt"] = natl_cov           # counterfactual uniform
    res_uniform = simulate_fast(rma_uniform, paths, cv, aph_window=10)

    out = {
        "selection_test": {"beta_stress": float(beta_stress), "se": float(se_stress),
                            "p_value": float(p), "n_county_crop": int(len(df)),
                            "coverage_pp_per_10pp_decline": float(effect_10pp * 100),
                            "interpretation": "positive beta => climate-stressed counties elect higher coverage"},
        "national_acreage_weighted_coverage": natl_cov,
        "transfer_actual_coverage_B": res_actual["tay"]["xsub_B"],
        "transfer_uniform_coverage_B": res_uniform["tay"]["xsub_B"],
        "endogeneity_effect_on_transfer_B": res_actual["tay"]["xsub_B"] - res_uniform["tay"]["xsub_B"],
        "residual_actual_B": res_actual["tay"]["total_B"],
        "residual_uniform_B": res_uniform["tay"]["total_B"],
    }
