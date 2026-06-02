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
