"""Revision: insurance mispricing with a REAL-TIME ROLLING APH simulation.

Reviewer 2 (Major Concerns 2 & 3) showed the original mispricing figure
($5.9 B yr-1) overstates the policy-relevant number because it compared a
forward 2040-2050 yield projection against a FROZEN historical APH baseline
(src/08_insurance.py uses yield_baseline as a constant). In reality, Actual
Production History (APH) is a 4-to-10-year rolling mean that updates every year,
so it mechanically absorbs most of a smooth climate trend at a ~5-year lag.

This script rebuilds the estimate to answer the reviewer precisely:

  (R2 #3a) Simulate the APH update mechanism: compare the forward projected
           yield against an APH-equivalent rolling mean computed in real time
           as years progress, instead of a frozen baseline.

  (R2 #3b) Report mispricing net of Trend-Adjusted Yield (TAY) endorsements at
           current participation, and bound the Yield-Exclusion (YE) effect.

  (R2 #3c) Decompose the gross gap into:
             (i)   what the rolling APH window absorbs mechanically,
             (ii)  what TAY absorbs at current participation,
             (iii) the RESIDUAL that forward-looking reform would actually
                   eliminate  --> the policy-relevant headline.

  (R2 #2)  Use the ACREAGE-WEIGHTED coverage election from the RMA Summary of
           Business (not a uniform 75%), report a coverage-level sensitivity,
           and report the Revenue-Protection vs Yield-Protection acreage mix
           (RP is the dominant product, so the analytical put captures the
           yield channel; the RP price channel is climate-neutral in
           expectation).

All dollars 2023 USD. Seed 42. Writes only to results/revision/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls  # noqa: F401 (kept for parity with pipeline env)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
OUT = PROJECT_ROOT / "results" / "revision"
OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# RMA crop_name -> model crop
RMA_CROP_MAP = {
    "CORN": "corn", "SOYBEANS": "soybeans", "WHEAT": "wheat_winter",
    "COTTON": "cotton", "GRAIN SORGHUM": "sorghum", "BARLEY": "barley",
    "OATS": "oats",
}
# Flat real prices (2023 USD) -- price largely cancels in the EI ratio, but we
# use a single consistent value per crop. These are 30-yr real averages
# (consistent with the DCF real-price fix); kept here so the module is standalone.
PRICE = {
    "corn": 4.10, "soybeans": 10.30, "wheat_winter": 5.70, "wheat_spring": 6.10,
    "cotton": 0.70, "sorghum": 4.00, "barley": 4.80, "oats": 3.10,
}
LOADING = 1.15
MAX_RATIO = 5.0
APH_WINDOW = 10           # RMA APH uses up to 10 years
TAY_LAG_YEARS = 5.5       # mean age of APH years -> TAY adds trend*lag
# TAY participation by crop (share of insured acres carrying a trend-adjusted
# endorsement). Corn/soy in the Corn Belt have high uptake; small grains lower.
# Documented as a parameter; sensitivity reported in the summary.
TAY_PARTICIPATION = {
    "corn": 0.55, "soybeans": 0.55, "wheat_winter": 0.35, "wheat_spring": 0.35,
    "sorghum": 0.30, "barley": 0.30, "oats": 0.20, "cotton": 0.30,
}
WINDOW = (2040, 2050)     # headline window (matches original $5.9B definition)


def expected_indemnity(K, mu, sigma):
    """E[max(K - X, 0)] for X ~ N(mu, sigma^2): analytical revenue put.

    Args:
        K: revenue guarantee, $/acre.
        mu: expected revenue, $/acre.
        sigma: revenue standard deviation, $/acre.
    Returns:
        Expected indemnity per acre ($, >= 0).
    """
    sigma = np.maximum(sigma, 1.0)
    z = (K - mu) / sigma
    return np.maximum((K - mu) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z), 0.0)


def build_rma_county_crop():
    """Aggregate RMA SOB (2014-2023) to county-crop with acreage-weighted
    coverage, RP/YP acre shares, observed premium per acre, and insured acres.

    The 2014-2023 window reflects post-2014-Farm-Bill coverage elections
    (Reviewer 2: elections shifted upward to 80-85% in the upper Corn Belt).

    Returns:
        DataFrame [fips, crop, cov_wt, rp_share, yp_share, prem_per_acre,
                   insured_acres].
    """
    rma = pd.read_parquet(
        DATA_RAW / "rma" / "rma_sob_all_years.parquet",
        columns=["year", "fips", "crop_name", "plan_code", "coverage_level",
                 "acres", "total_premium", "indemnity"],
    )
    rma = rma[rma["year"].between(2014, 2023)].copy()
    rma["fips"] = rma["fips"].astype(str).str.zfill(5)
    rma["crop"] = rma["crop_name"].str.strip().str.upper().map(RMA_CROP_MAP)
    rma = rma[rma["crop"].notna()].copy()
    rma["cl"] = pd.to_numeric(rma["coverage_level"], errors="coerce")
    rma = rma[(rma["acres"] > 0) & rma["cl"].between(0.45, 0.95)].copy()

    # Revenue plans: 02 RP, 03 RPHPE, 32/33 SCO-RP, 25 RA, 44 CRC. Yield: 01 YP, 31 SCO-YP.
    rev_plans = {"02", "03", "25", "44", "32", "33"}
    rma["is_rev"] = rma["plan_code"].astype(str).str.zfill(2).isin(rev_plans)

    g = rma.groupby(["fips", "crop"])
    out = pd.DataFrame({
        "insured_acres": g["acres"].sum() / rma.groupby(["fips", "crop"])["year"].nunique(),
        "cov_wt": g.apply(lambda d: np.average(d["cl"], weights=d["acres"]), include_groups=False),
        "rp_acres": rma[rma.is_rev].groupby(["fips", "crop"])["acres"].sum(),
        "tot_acres": g["acres"].sum(),
        "prem_sum": g["total_premium"].sum(),
        "acre_sum": g["acres"].sum(),
    }).reset_index()
    out["rp_acres"] = out["rp_acres"].fillna(0.0)
    out["rp_share"] = out["rp_acres"] / out["tot_acres"].replace(0, np.nan)
    out["yp_share"] = 1.0 - out["rp_share"]
    out["prem_per_acre"] = out["prem_sum"] / out["acre_sum"].replace(0, np.nan)
    out = out[out["insured_acres"] > 0].dropna(subset=["cov_wt", "prem_per_acre"])
    return out[["fips", "crop", "cov_wt", "rp_share", "yp_share",
                "prem_per_acre", "insured_acres"]]


def build_yield_paths():
    """Realized yield path per county-crop: observed (<=2024) + projected (2025-2050).

    Returns:
        (paths, cv) where paths is a long DataFrame [fips, crop, year, y] and
        cv is [fips, crop, cv] interannual coefficient of variation from
        observed 2008-2023 yields.
    """
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet",
                         columns=["fips", "year", "crop", "yield_bu_acre"])
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    obs = fm[(fm["year"] <= 2024) & (fm["yield_bu_acre"] > 0)].rename(
        columns={"yield_bu_acre": "y"})[["fips", "crop", "year", "y"]]

    proj = pd.read_parquet(PROJECTIONS_DIR / "yield_projections_SSP245.parquet",
                           columns=["fips", "year", "crop", "yield_projected"])
    proj["fips"] = proj["fips"].astype(str).str.zfill(5)
    proj = proj[proj["year"] >= 2025].rename(columns={"yield_projected": "y"})
    proj = proj[["fips", "crop", "year", "y"]]
    proj["y"] = proj["y"].clip(lower=0.0)

    paths = pd.concat([obs, proj], ignore_index=True).sort_values(
        ["fips", "crop", "year"]).reset_index(drop=True)

    rec = obs[obs["year"].between(2008, 2023)]
    cv = (rec.groupby(["fips", "crop"])["y"]
          .agg(["mean", "std", "count"]).reset_index())
    cv = cv[cv["count"] >= 5]
    cv["cv"] = (cv["std"] / cv["mean"]).clip(0.05, 0.50).fillna(0.20)
    return paths, cv[["fips", "crop", "cv"]]


def simulate(rma, paths, cv):
    """Simulate mispricing under frozen / rolling / rolling+TAY APH for each
    projection year, anchored to observed premiums via the EI ratio.

    For each county-crop and year t, we hold the revenue guarantee at
    K = APH_method_t * coverage * price and compare the expected indemnity under
    the TRUE expected yield (realized path at t) against the expected indemnity
    the program implicitly prices (yield centered at APH_method_t). Mispricing
    per acre = observed_premium * (EI_true / EI_priced - 1). Price and CV
    largely cancel in the ratio, so the result is robust to price level.

    Returns:
        (per_year_df, county_year_df). per_year_df has total mispricing and
        cross-subsidy by year and method.
    """
    # pivot realized path to wide for fast trailing-window ops per county-crop
    paths = paths.merge(cv, on=["fips", "crop"], how="inner")
    paths = paths.merge(rma[["fips", "crop", "cov_wt", "prem_per_acre",
                             "insured_acres"]], on=["fips", "crop"], how="inner")

    # baseline mean yield (pre-2025 observed mean) used for sigma and frozen APH
    base = (paths[paths["year"] <= 2024].groupby(["fips", "crop"])["y"]
            .mean().rename("aph_frozen").reset_index())
    paths = paths.merge(base, on=["fips", "crop"], how="left")

    records = []
    for (fips, crop), d in paths.groupby(["fips", "crop"], sort=False):
        d = d.sort_values("year")
        yr = d["year"].values
        y = d["y"].values
        price = PRICE.get(crop, 4.0)
        cvv = d["cv"].iloc[0]
        cov = float(d["cov_wt"].iloc[0])
        prem = float(d["prem_per_acre"].iloc[0])
        acres = float(d["insured_acres"].iloc[0])
        aph_frozen = float(d["aph_frozen"].iloc[0])
        if not np.isfinite(aph_frozen) or aph_frozen <= 0:
            continue
        sigma = aph_frozen * cvv * price          # fixed interannual revenue risk
        ptay = TAY_PARTICIPATION.get(crop, 0.3)

        for t in range(WINDOW[0], WINDOW[1] + 1):
            mask_win = (yr >= t - APH_WINDOW) & (yr <= t - 1)
            if mask_win.sum() < 4:
                continue
            roll = float(np.mean(y[mask_win]))
            # county yield trend over the trailing window (RMA TAY uses a linear trend)
            yy = y[mask_win]; xx = yr[mask_win]
            slope = np.polyfit(xx, yy, 1)[0] if mask_win.sum() >= 4 else 0.0
            aph_tay = roll + ptay * slope * TAY_LAG_YEARS
            true_y = float(y[yr == t][0]) if (yr == t).any() else roll

            def mp(aph):
                K = aph * cov * price
                ei_true = expected_indemnity(K, true_y * price, sigma)
                ei_priced = expected_indemnity(K, aph * price, sigma)
                if ei_priced < 1e-6:
                    ratio = 1.0 if ei_true < 1e-6 else MAX_RATIO
                else:
                    ratio = min(ei_true / ei_priced, MAX_RATIO)
                return prem * (ratio - 1.0) * acres   # annual $ flow

            records.append({
                "fips": fips, "crop": crop, "year": t,
                "flow_frozen": mp(aph_frozen),
                "flow_roll": mp(roll),
                "flow_tay": mp(aph_tay),
                "acres": acres,
            })

    cy = pd.DataFrame(records)

    def agg(col):
        up = cy[cy[col] > 0].groupby("year")[col].sum()
        ov = cy[cy[col] < 0].groupby("year")[col].sum().abs()
        return pd.DataFrame({f"{col}_under": up, f"{col}_over": ov}).fillna(0.0)

    per_year = pd.concat([agg("flow_frozen"), agg("flow_roll"),
                          agg("flow_tay")], axis=1).fillna(0.0)
    for m in ["frozen", "roll", "tay"]:
        per_year[f"{m}_total"] = per_year[f"flow_{m}_under"] + per_year[f"flow_{m}_over"]
        per_year[f"{m}_xsub"] = per_year[[f"flow_{m}_under", f"flow_{m}_over"]].min(axis=1)
    return per_year.reset_index(), cy


def coverage_sensitivity(rma, paths, cv, levels=(0.55, 0.65, 0.75, 0.85)):
    """Residual (rolling+TAY) mispricing under fixed coverage levels."""
    res = {}
    for L in levels:
        r2 = rma.copy(); r2["cov_wt"] = L
        py, _ = simulate(r2, paths, cv)
        win = py[py["year"].between(*WINDOW)]
        res[L] = float(win["tay_total"].mean() / 1e9)
    return res


def main():
    print("Building RMA county-crop coverage/plan aggregates (2014-2023)...")
    rma = build_rma_county_crop()
