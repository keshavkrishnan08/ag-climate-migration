"""
Fix 4: IV estimation of farm income -> outmigration elasticity.

Approach: Two-stage least squares with two-way county + year FE.

Instrument: Weather-driven income shock. For each county-year, we compute:
    Z_it = sum_c [ yield_detrended_ict * acres_ic_bar * price_c ] / baseline_income_i
    where yield_detrended = actual yield minus county-crop quadratic trend,
    acres_ic_bar = county-crop mean acres (fixed exposure),
    price_c = national commodity price.
    This isolates the weather-driven component of farm revenue.

Treatment: Farm income deviation from county baseline.
    D_it = (income_it - income_i_bar) / income_i_bar
    where income_it = sum_c [ yield_ict * acres_ict * price_c * deflator ]

Outcome: Net outmigration rate from population change.
    Y_it = -(pop_t - pop_{t-1}) / pop_{t-1}
    Positive = population loss = net outmigration.

Sample: Rural Corn Belt counties (pop < 50,000), 2010-2023 (ACS range).
FE:     County + year (absorbed via two-way demeaning).
SE:     Cluster-robust at county level.
Gate:   First-stage F > 10.

Uses manual 2SLS via numpy/scipy (statsmodels has scipy compat issue).

Author: Keshav Krishnan
Date:   2026-03-17
"""
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
CORN_BELT_STATE_FIPS = [
    "19",  # Iowa
    "17",  # Illinois
    "18",  # Indiana
    "39",  # Ohio
    "27",  # Minnesota
    "55",  # Wisconsin
    "31",  # Nebraska
    "29",  # Missouri
    "46",  # South Dakota
    "38",  # North Dakota
    "20",  # Kansas
]

POP_CAP = 50_000  # rural threshold

# Commodity prices in 2023 USD (5-year average, $/bushel except cotton $/lb)
COMMODITY_PRICES = {
    "corn": 5.50,
    "soybeans": 12.80,
    "wheat_winter": 7.20,
    "wheat_spring": 8.10,
    "cotton": 0.78,      # $/lb
    "sorghum": 5.30,
    "barley": 6.10,
    "oats": 3.80,
}

CPI_2023 = 304.703  # from data/raw/other/cpi_annual.csv

RANDOM_SEED = 42

# Min yield thresholds to filter silage/cover crop records
MIN_YIELD = {
    "corn": 50,
    "soybeans": 10,
    "wheat_winter": 10,
    "wheat_spring": 10,
    "sorghum": 15,
    "barley": 10,
    "oats": 10,
    "cotton": 100,  # lbs/acre
}


def ols_fit(y, X):
    """Fit OLS via numpy least squares.

    Args:
        y: Response vector (n,).
        X: Design matrix (n, k). Should include constant if desired.

    Returns:
        Dict with coefficients, residuals, fitted values, and diagnostics.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n, k = X.shape

    beta, _, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    fitted = X @ beta
    resid = y - fitted
    ss_res = resid @ resid
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    sigma2 = ss_res / (n - k)
    var_beta = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(var_beta))
    t_stats = beta / se

    return {
        "beta": beta,
        "fitted": fitted,
        "residuals": resid,
        "se": se,
        "t_stats": t_stats,
        "r_squared": r_squared,
        "sigma2": sigma2,
        "n": n,
        "k": k,
    }


def load_and_clean_yields():
    """Load NASS yields, deduplicate, and filter to Corn Belt.

    Returns:
        pd.DataFrame with [fips, year, crop, yield_bu_acre, acres_harvested,
        production, state_fips].
    """
    yields = pd.read_parquet(
        PROJECT_ROOT / "data/raw/nass/nass_county_yields.parquet",
        columns=["fips", "year", "crop", "yield_bu_acre", "acres_harvested", "production"],
    )
    yields["state_fips"] = yields["fips"].str[:2]
    yields = yields[yields["state_fips"].isin(CORN_BELT_STATE_FIPS)].copy()

    # Filter to crops we have prices for
    yields = yields[yields["crop"].isin(COMMODITY_PRICES.keys())].copy()

    # Remove silage/cover crop records using min yield thresholds
    mask = pd.Series(False, index=yields.index)
    for crop, min_y in MIN_YIELD.items():
        crop_mask = (yields["crop"] == crop) & (yields["yield_bu_acre"] >= min_y)
        mask = mask | crop_mask
    yields = yields[mask].copy()

    # Deduplicate: take the record with max production per fips/year/crop
    yields = (
        yields.sort_values("production", ascending=False)
        .groupby(["fips", "year", "crop"])
        .first()
        .reset_index()
    )

    return yields


def detrend_yields(yields):
    """Remove quadratic time trend from yields within each county-crop.

    Detrending isolates the weather-driven component. The residual
    (yield minus trend) captures weather shocks -- exactly what we need
    for the instrument.

    Args:
        yields: DataFrame with [fips, year, crop, yield_bu_acre].

    Returns:
        DataFrame with added columns: yield_trend, yield_detrended.
    """
    results = []
    for (fips, crop), grp in yields.groupby(["fips", "crop"]):
        if len(grp) < 5:
            continue
        grp = grp.sort_values("year").copy()
        t = grp["year"].values - grp["year"].values[0]
        try:
            coeffs = np.polyfit(t, grp["yield_bu_acre"].values, 2)
            grp["yield_trend"] = np.polyval(coeffs, t)
            grp["yield_detrended"] = grp["yield_bu_acre"] - grp["yield_trend"]
        except (np.linalg.LinAlgError, ValueError):
            continue
        results.append(grp)

    return pd.concat(results, ignore_index=True)


def build_iv_panel(yields, cpi):
    """Construct the county-year panel with treatment and instrument.

    Treatment (D): Farm income deviation from county mean, in fractional units.
        D_it = (income_it - income_i_bar) / income_i_bar

    Instrument (Z): Weather-driven income shock, using FIXED acreage weights.
        Z_it = sum_c[ yield_detrended_ict * acres_ic_bar * price_c ] / baseline_income_i
        Using fixed (mean) acreage prevents endogenous crop switching from
        contaminating the instrument.

    Args:
        yields: DataFrame with detrended yields.
        cpi: DataFrame with [year, cpi].

    Returns:
        DataFrame at county-year level with farm_income_deviation, weather_income_shock.
    """
    yields = yields.merge(cpi[["year", "cpi"]], on="year", how="left")
    deflator = CPI_2023 / yields["cpi"]

    yields["price"] = yields["crop"].map(COMMODITY_PRICES)

    # Revenue per crop-county-year (actual)
    yields["revenue_2023usd"] = (
        yields["yield_bu_acre"] * yields["acres_harvested"] * yields["price"] * deflator
    )

    # Fixed acreage weights for the instrument: county-crop mean acres
    # This prevents endogenous crop switching from contaminating Z
    mean_acres = (
        yields.groupby(["fips", "crop"])["acres_harvested"]
        .mean()
        .rename("acres_mean")
    )
    yields = yields.merge(mean_acres, on=["fips", "crop"], how="left")

    # Weather component of revenue (instrument numerator)
    # Uses detrended yield (weather only) and FIXED acreage
    yields["weather_revenue_2023usd"] = (
        yields["yield_detrended"] * yields["acres_mean"] * yields["price"] * deflator
    )

    # Aggregate to county-year
    county_year = (
        yields.groupby(["fips", "year"])
        .agg(
            farm_income_proxy=("revenue_2023usd", "sum"),
            weather_revenue=("weather_revenue_2023usd", "sum"),
            total_acres=("acres_harvested", "sum"),
            n_crops=("crop", "count"),
        )
        .reset_index()
    )

    # Baseline income: county mean across all years
    baseline = county_year.groupby("fips")["farm_income_proxy"].mean().rename("baseline_income")
    county_year = county_year.merge(baseline, on="fips", how="left")

    # Treatment: fractional deviation from county mean
    county_year["farm_income_deviation"] = (
        (county_year["farm_income_proxy"] - county_year["baseline_income"])
        / county_year["baseline_income"]
    )

    # Instrument: weather shock as fraction of baseline income
    county_year["weather_income_shock"] = (
        county_year["weather_revenue"] / county_year["baseline_income"]
    )

    # Quality filters
    county_year = county_year[county_year["baseline_income"] > 1_000_000].copy()
    county_counts = county_year.groupby("fips").size()
    good_counties = county_counts[county_counts >= 10].index
    county_year = county_year[county_year["fips"].isin(good_counties)].copy()

    return county_year


def load_migration_outcome():
    """Load ACS demographics and compute migration outcome variables.

    ACS B07001 column mislabeling — confirmed by inspection (2026-03-18):
        'moved_diff_county_same_state' = B07001_002E (Same house 1yr ago, ~87% of total)
        'moved_diff_state'             = B07001_049E (Moved diff county, same state, ~6%)
        'moved_from_abroad'            = B07001_065E (Moved diff state, ~2%)

    The download script pulled variables in shifted order, so names are wrong.
    We rename them to their true content and build two outcomes:

        Spec A: net_outmigration_rate = -(pop_t - pop_{t-1}) / pop_{t-1}
            Positive = population loss. Noisy (births/deaths confound migration).

        Spec C (primary fix): true_diff_county_in_rate
            = B07001_049E (true diff-county same-state in-movers) / total_population
            These are actual inter-county in-migrants, cleanly measured.
            Expected direction: higher income -> more in-migration -> positive beta.
            Equivalently: lower income -> less in-migration.

        Spec D: long_distance_in_rate
            = (true_diff_county + true_diff_state) / total_population
            Captures both inter-county and inter-state in-migration.

    Returns:
        DataFrame with [fips, year, outmigration_rate, true_diff_county_in_rate,
                         long_distance_in_rate, intercounty_inmigration_rate,
                         total_population].
    """
    mig = pd.read_parquet(
        PROJECT_ROOT / "data/raw/census/acs_migration_data.parquet",
        columns=["fips", "year", "mobility_total", "moved_diff_county_same_state",
                 "moved_diff_state", "moved_from_abroad"],
    )
    demo = pd.read_parquet(
        PROJECT_ROOT / "data/raw/census/acs_county_demographics.parquet",
        columns=["fips", "year", "total_population", "median_household_income"],
    )

    # Rename to true ACS variable content (correcting mislabeling)
    mig = mig.rename(columns={
        "moved_diff_county_same_state": "true_same_house",     # B07001_002E
        "moved_diff_state": "true_diff_county_same_state",     # B07001_049E
        "moved_from_abroad": "true_diff_state",                # B07001_065E
    })

    # Primary outcome (Spec A): net outmigration from population change
    demo = demo.sort_values(["fips", "year"]).copy()
    demo["pop_change_rate"] = demo.groupby("fips")["total_population"].pct_change()
    demo["outmigration_rate"] = -demo["pop_change_rate"]

    panel = mig.merge(demo, on=["fips", "year"], how="inner")

    # Spec C (primary fix): true inter-county in-migration rate
    # Uses B07001_049E: moved from different county, same state
    panel["true_diff_county_in_rate"] = (
        panel["true_diff_county_same_state"].fillna(0)
        / panel["total_population"].replace(0, np.nan)
    )

    # Spec D: long-distance in-migration (diff county + diff state)
    panel["long_distance_in_rate"] = (
        (panel["true_diff_county_same_state"].fillna(0)
         + panel["true_diff_state"].fillna(0))
        / panel["total_population"].replace(0, np.nan)
    )

    # Legacy Spec B rate (kept for continuity): all movers / total pop
    # = (mobility_total - same_house) / total_population
    panel["intercounty_inmigration_rate"] = (
        (panel["mobility_total"] - panel["true_same_house"]).clip(lower=0)
        / panel["total_population"].replace(0, np.nan)
    )
