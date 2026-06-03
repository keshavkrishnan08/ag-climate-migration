"""Revision: restrict the rural-decline analysis to farming-dependent counties
and replace coincident indicator COUNTS with MARGINAL EFFECTS (Reviewer 1 #2, Fig 7B).

Reviewer 1 raised three linked points:
  (a) The yield -> outmigration mechanism may be externally invalid for the
      diversified US economy; only counties with a high farm share of income
      should be expected to show it. Restrict to that subset.
  (b) Figure 7B (coincident counts of decline indicators) is misleading because
      the indicators are merely concurrent; report the MARGINAL effect of yield
      decline on each indicator instead.
  (c) The weather IV likely violates the exclusion restriction (weather affects
      migration via amenity / winter-mildness channels, Rappaport 2007), so the
      defensible object is the direct chain yield -> farm income -> local fiscal
      capacity, not a causal migration elasticity.

This script:
  1. Flags the 444 ERS farming-dependent counties (Type_2015_Farming_NO=1).
  2. Recomputes the share of counties with >=4 decline indicators within the
     farming-dependent subset vs the rest (the restricted observational result).
  3. Estimates the marginal effect of a 1-SD adverse yield anomaly on each
     decline outcome (population growth, income growth, net outmigration) with
     two-way (county + year) fixed effects, on the farming-dependent subset.
  4. Re-estimates a reduced-form / OLS yield->outmigration relationship on the
     subset and reports it honestly, with the exclusion-restriction caveat.

Seed 42. Writes only to results/revision/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PUB = ROOT / "data" / "published_dataset"
OUT = ROOT / "results" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(42)


def farming_dependent():
    cc = pd.read_csv(DATA_RAW / "other" / "ers_atlas" / "CountyClassifications.csv",
                     dtype=str, encoding="latin-1")
    cc = cc.rename(columns={cc.columns[0]: "fips"})
    cc["fips"] = cc["fips"].str.zfill(5)
    fd = cc[cc["Attribute"] == "Type_2015_Farming_NO"][["fips", "Value"]]
    fd["farm_dependent"] = (fd["Value"] == "1").astype(int)
    return fd[["fips", "farm_dependent"]]


def demean_twoway(df, cols, ent="fips", time="year"):
    """Two-way within transform (county + year FE) by iterated demeaning."""
    out = df.copy()
    for c in cols:
        s = out[c].astype(float)
        for _ in range(20):
            s = s - s.groupby(out[ent]).transform("mean")
            s = s - s.groupby(out[time]).transform("mean")
        out[c + "_dm"] = s
    return out


def ols(y, X):
    """OLS with HC1 SE. X already includes intercept column if desired."""
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    n, k = X.shape
    XtX_inv = np.linalg.inv(XtX)
    # HC1
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    cov = XtX_inv @ S @ XtX_inv * (n / (n - k))
    se = np.sqrt(np.diag(cov))
    from scipy import stats
    t = beta / se
    p = 2 * (1 - stats.norm.cdf(np.abs(t)))
    return beta, se, p


def build_panel():
    """County-year panel: acreage-weighted yield anomaly + ACS pop/income/migration."""
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet",
                         columns=["fips", "year", "crop", "yield_anomaly",
                                  "acres_harvested"])
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    fm = fm.dropna(subset=["yield_anomaly"])
    # acreage-weighted county yield anomaly (vectorized)
    fm["w"] = fm["acres_harvested"].clip(lower=0).fillna(0)
    fm["wy"] = fm["yield_anomaly"] * fm["w"]
    agg = fm.groupby(["fips", "year"]).agg(
        sumwy=("wy", "sum"), sumw=("w", "sum"),
        meany=("yield_anomaly", "mean")).reset_index()
    agg["yield_anom"] = np.where(agg["sumw"] > 0, agg["sumwy"] / agg["sumw"].replace(0, np.nan),
                                 agg["meany"])
    cy = agg[["fips", "year", "yield_anom"]]

    demo = pd.read_parquet(DATA_RAW / "census" / "acs_county_demographics.parquet",
                           columns=["fips", "year", "total_population",
                                    "median_household_income"])
    demo = demo.rename(columns={"total_population": "population",
                                "median_household_income": "median_income"})
    demo["fips"] = demo["fips"].astype(str).str.zfill(5)
    demo = demo.sort_values(["fips", "year"])
    demo["pop_growth"] = demo.groupby("fips")["population"].pct_change()
    demo["inc_growth"] = (np.log(demo["median_income"])
                          - np.log(demo.groupby("fips")["median_income"].shift(1)))

    mig = pd.read_parquet(DATA_RAW / "census" / "acs_migration_data.parquet")
    mig["fips"] = mig["fips"].astype(str).str.zfill(5)
    migcols = [c for c in mig.columns if "moved" in c or "mobility" in c]
    keep = ["fips", "year"] + migcols
    mig = mig[keep]

    panel = (cy.merge(demo[["fips", "year", "pop_growth", "inc_growth", "population"]],
                      on=["fips", "year"], how="inner")
               .merge(farming_dependent(), on="fips", how="left"))
    panel["farm_dependent"] = panel["farm_dependent"].fillna(0).astype(int)
    # lag yield anomaly one year (decline follows the shock)
    panel = panel.sort_values(["fips", "year"])
    panel["yield_anom_lag1"] = panel.groupby("fips")["yield_anom"].shift(1)
    return panel


def marginal_effects(panel, subset_mask, label):
    """Two-way FE marginal effect of lagged yield anomaly on decline outcomes."""
    res = {}
    d = panel[subset_mask].dropna(subset=["yield_anom_lag1"]).copy()
    for outcome in ["pop_growth", "inc_growth"]:
        dd = d.dropna(subset=[outcome]).copy()
        dd = dd[np.isfinite(dd[outcome]) & np.isfinite(dd["yield_anom_lag1"])]
        if len(dd) < 200:
            continue
        dm = demean_twoway(dd, [outcome, "yield_anom_lag1"])
        y = dm[outcome + "_dm"].values
        X = dm[["yield_anom_lag1_dm"]].values
        X = np.column_stack([np.ones(len(X)), X])
        beta, se, p = ols(y, X)
        sd_shock = dd["yield_anom_lag1"].std()
        res[outcome] = {
            "beta_per_unit": float(beta[1]), "se": float(se[1]), "p": float(p[1]),
            "marginal_per_1sd_yield_drop_pct": float(-beta[1] * sd_shock * 100),
            "n": int(len(dd)), "n_counties": int(dd["fips"].nunique()),
        }
    res["label"] = label
    return res


def main():
    fd = farming_dependent()
    di = pd.read_csv(PUB / "county_decline_indicators.csv", dtype={"fips": str})
    di["fips"] = di["fips"].str.zfill(5)
    di = di.merge(fd, on="fips", how="left")
    di["farm_dependent"] = di["farm_dependent"].fillna(0).astype(int)

    # Restricted observational result
    n_fd = int(di["farm_dependent"].sum())
    fd_4plus = int(((di["farm_dependent"] == 1) & (di["n_decline_indicators"] >= 4)).sum())
    nonfd_4plus = int(((di["farm_dependent"] == 0) & (di["n_decline_indicators"] >= 4)).sum())
    fd_share = fd_4plus / max(n_fd, 1)
    nonfd_n = int((di["farm_dependent"] == 0).sum())
    nonfd_share = nonfd_4plus / max(nonfd_n, 1)

    print(f"Farming-dependent counties in indicator panel: {n_fd}")
    print(f"  >=4 indicators (farm-dependent):  {fd_4plus}  ({fd_share*100:.1f}%)")
    print(f"  >=4 indicators (other counties):  {nonfd_4plus}  ({nonfd_share*100:.1f}%)")
    print(f"  enrichment ratio: {fd_share/max(nonfd_share,1e-9):.2f}x")

    print("\nBuilding county-year panel for marginal effects...")
    panel = build_panel()
    me_fd = marginal_effects(panel, panel["farm_dependent"] == 1, "farming_dependent")
    me_all = marginal_effects(panel, panel["farm_dependent"].notna(), "all_counties")

    print("\nMarginal effect of a 1-SD adverse yield anomaly (two-way FE):")
    for o in ["pop_growth", "inc_growth"]:
        if o in me_fd:
            r = me_fd[o]
            print(f"  [farm-dep] {o}: {r['marginal_per_1sd_yield_drop_pct']:+.3f} pct "
                  f"(beta={r['beta_per_unit']:+.4f}, p={r['p']:.3f}, n={r['n']})")

    summary = {
        "n_farming_dependent_counties": n_fd,
        "decline_4plus_farm_dependent": fd_4plus,
        "decline_4plus_share_farm_dependent": fd_share,
        "decline_4plus_other": nonfd_4plus,
        "decline_4plus_share_other": nonfd_share,
        "enrichment_ratio": fd_share / max(nonfd_share, 1e-9),
