"""Vectorized insurance mispricing simulation + round-2 robustness battery.

Reimplements the rolling-APH decomposition (insurance_rolling_aph.py) without the
per-county Python loop, so we can sweep robustness dimensions a tough reviewer
would request next:
  * APH window length (4 / 7 / 10 years) -- shorter windows absorb the trend
    faster, reducing the residual.
  * Yield-Exclusion (YE) at participation -- drop the worst year in the window,
    raising APH in disaster-prone counties (works against TAY).
  * Climate scenario (SSP2-4.5 vs SSP3-7.0).
  * Per-crop decomposition.

A vectorized re-implementation also cross-checks the headline produced by the
slow loop. Seed 42; writes only to results/revision/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from insurance_rolling_aph import (build_rma_county_crop, PRICE, LOADING, MAX_RATIO,
                                    TAY_PARTICIPATION, TAY_LAG_YEARS)

DATA_PROCESSED = ROOT / "data" / "processed"
PROJ = ROOT / "data" / "projections"
OUT = ROOT / "results" / "revision"
np.random.seed(42)
WIN = (2040, 2050)


def expected_indemnity_vec(K, mu, sigma):
    sigma = np.maximum(sigma, 1.0)
    z = (K - mu) / sigma
    return np.maximum((K - mu) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z), 0.0)


def build_paths(scenario):
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet",
                         columns=["fips", "year", "crop", "yield_bu_acre"])
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    obs = fm[(fm["year"] <= 2024) & (fm["yield_bu_acre"] > 0)].rename(
        columns={"yield_bu_acre": "y"})[["fips", "crop", "year", "y"]]
    pj = pd.read_parquet(PROJ / f"yield_projections_{scenario}.parquet",
                         columns=["fips", "year", "crop", "yield_projected"])
    pj["fips"] = pj["fips"].astype(str).str.zfill(5)
    pj = pj[pj["year"] >= 2025].rename(columns={"yield_projected": "y"})[["fips", "crop", "year", "y"]]
    pj["y"] = pj["y"].clip(lower=0)
    paths = pd.concat([obs, pj], ignore_index=True)
    rec = obs[obs["year"].between(2008, 2023)]
    cv = rec.groupby(["fips", "crop"])["y"].agg(["mean", "std", "count"]).reset_index()
    cv = cv[cv["count"] >= 5]
    cv["cv"] = (cv["std"] / cv["mean"]).clip(0.05, 0.50).fillna(0.20)
    return paths, cv[["fips", "crop", "cv"]]


def simulate_fast(rma, paths, cv, aph_window=10, ye=False, ye_participation=0.0):
    """Vectorized decomposition. Returns dict of $B/yr averaged over WIN."""
    keys = rma[["fips", "crop"]].drop_duplicates()
    paths = paths.merge(keys, on=["fips", "crop"], how="inner")
    wide = paths.pivot_table(index=["fips", "crop"], columns="year", values="y", aggfunc="first")
    years_needed = list(range(WIN[0] - aph_window, WIN[1] + 1))
    for y in years_needed:
        if y not in wide.columns:
            wide[y] = np.nan
    wide = wide.sort_index(axis=1)

    meta = (rma.set_index(["fips", "crop"])
            .reindex(wide.index)[["cov_wt", "prem_per_acre", "insured_acres"]])
    cvv = (cv.set_index(["fips", "crop"]).reindex(wide.index)["cv"]).fillna(0.20).values
    # frozen APH = pre-2025 observed mean
    obs_cols = [c for c in wide.columns if c <= 2024]
    aph_frozen = wide[obs_cols].mean(axis=1).values
    price = pd.Index(wide.index.get_level_values("crop")).map(
        lambda c: PRICE.get(c, 4.0)).to_numpy(dtype=float)
    ptay = pd.Index(wide.index.get_level_values("crop")).map(
        lambda c: TAY_PARTICIPATION.get(c, 0.3)).to_numpy(dtype=float)
    cov = meta["cov_wt"].values
    prem = meta["prem_per_acre"].values
    acres = meta["insured_acres"].values
    sigma = aph_frozen * cvv * price
    valid = np.isfinite(aph_frozen) & (aph_frozen > 0) & np.isfinite(cov) & np.isfinite(prem)

    flows = {"frozen": [], "roll": [], "tay": []}
    state = pd.Index(wide.index.get_level_values("fips")).str[:2].to_numpy()
    rows = []
    for T in range(WIN[0], WIN[1] + 1):
        wcols = [y for y in range(T - aph_window, T)]
        W = wide[wcols].values  # (n, window)
        roll = np.nanmean(W, axis=1)
        if ye:
            # drop worst year in window for participating share; vectorize blended APH
            Wsort = np.sort(W, axis=1)
            roll_ye = np.nanmean(Wsort[:, 1:], axis=1)  # exclude minimum
            roll = (1 - ye_participation) * roll + ye_participation * roll_ye
        # trailing slope (OLS) vectorized
        x = np.array(wcols, dtype=float); xm = x.mean(); xc = x - xm
        denom = np.sum(xc ** 2)
        slope = np.nansum((W - np.nanmean(W, axis=1, keepdims=True)) * xc, axis=1) / denom
        aph_tay = roll + ptay * slope * TAY_LAG_YEARS
        true_y = wide[T].values if T in wide.columns else roll

        def mp(aph):
            K = aph * cov * price
            ei_true = expected_indemnity_vec(K, true_y * price, sigma)
