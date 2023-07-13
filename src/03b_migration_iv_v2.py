"""
Fix 4 (v2): IV estimation — farm income -> outmigration elasticity.
Three new specifications targeting cleaner out-migration measurement.

Approach 1: Net migration from population components
    net_out_migration_rate = -(pop_t - pop_{t-1}) / pop_{t-1}
    BUT cleaned:
      - Remove counties with |pop_change| > 10% (boundary changes)
      - 3-year rolling population change to smooth noise
      - Weighted by baseline population (upweight large counties)

Approach 2: Same-house inverse proxy
    mobility_rate = 1 - (same_house / total_population)
    = (mobility_total - same_house) / total_population
    Interpretation: fraction of current residents who moved into the county
    in the past year. Closely parallels Spec C/B but includes same-county movers.
    Expected sign: positive (income -> more in-migration -> higher mobility rate).
    Note: this is an IN-migration proxy, not pure out-migration, but captures
    the same economic channel. Lower income -> fewer arrivals AND more departures.

Approach 3: Gross out-migration via population accounting
    gross_out_rate = net_outmig_rate + in_migration_rate
    = -(pop_t - pop_{t-1})/pop_{t-1} + true_diff_county_in_rate
    Captures actual outflows regardless of inflow variation.

Summary of all IV specifications:
    Spec A  : net outmigration (raw pop change)              [original, p=0.49]
    Spec A2 : net outmigration cleaned (excl boundary, 3yr)  [Approach 1]
    Spec A3 : net outmigration pop-weighted                   [Approach 1 variant]
    Spec B  : gross mobility rate (1 - same_house/pop)        [Approach 2]
    Spec C  : true diff-county in-migration (B07001_049E)     [prior primary fix, p=0.019]
    Spec D  : long-distance in-migration (diff-county + diff-state)
    Spec E  : gross out-migration (net + in-migration)        [Approach 3]

Author: Keshav Krishnan
Date:   2026-03-18
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

POP_CAP = 50_000          # rural threshold
POP_CHANGE_EXTREME = 0.10 # exclude counties with |pop_change| > 10% (boundary changes)

# Commodity prices in 2023 USD (5-year average)
COMMODITY_PRICES = {
    "corn": 5.50,
    "soybeans": 12.80,
    "wheat_winter": 7.20,
    "wheat_spring": 8.10,
    "cotton": 0.78,
    "sorghum": 5.30,
    "barley": 6.10,
    "oats": 3.80,
}

CPI_2023 = 304.703

# Min yield thresholds to filter silage/cover crop records
MIN_YIELD = {
    "corn": 50,
    "soybeans": 10,
    "wheat_winter": 10,
    "wheat_spring": 10,
    "sorghum": 15,
    "barley": 10,
    "oats": 10,
    "cotton": 100,
}


# ──────────────────────────────────────────────────────────────────────
# OLS helpers
# ──────────────────────────────────────────────────────────────────────

def ols_fit(y, X):
    """Fit OLS via numpy least squares.

    Args:
        y: Response vector (n,).
        X: Design matrix (n, k).

    Returns:
        Dict with beta, fitted, residuals, se, t_stats, r_squared.
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


def demean_twoway(arr, entity_ids, time_ids):
    """Apply two-way demeaning (within transformation for FE).

    For variable x: x_tilde = x - x_bar_i - x_bar_t + x_bar

    Args:
        arr: 1D array of values.
        entity_ids: Entity identifiers.
        time_ids: Time period identifiers.

    Returns:
        Demeaned array.
    """
    arr = np.asarray(arr, dtype=np.float64)
    grand_mean = arr.mean()

    entity_uniq, entity_inv = np.unique(entity_ids, return_inverse=True)
    entity_sums = np.bincount(entity_inv, weights=arr)
    entity_counts = np.bincount(entity_inv)
    entity_means = entity_sums / entity_counts

    time_uniq, time_inv = np.unique(time_ids, return_inverse=True)
    time_sums = np.bincount(time_inv, weights=arr)
    time_counts = np.bincount(time_inv)
    time_means = time_sums / time_counts

    return arr - entity_means[entity_inv] - time_means[time_inv] + grand_mean


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_and_clean_yields():
    """Load NASS yields, deduplicate, and filter to Corn Belt.

    Returns:
        DataFrame with fips, year, crop, yield_bu_acre, acres_harvested,
        production, state_fips.
    """
    yields = pd.read_parquet(
        PROJECT_ROOT / "data/raw/nass/nass_county_yields.parquet",
        columns=["fips", "year", "crop", "yield_bu_acre", "acres_harvested", "production"],
    )
    yields["state_fips"] = yields["fips"].str[:2]
    yields = yields[yields["state_fips"].isin(CORN_BELT_STATE_FIPS)].copy()
    yields = yields[yields["crop"].isin(COMMODITY_PRICES.keys())].copy()

    mask = pd.Series(False, index=yields.index)
    for crop, min_y in MIN_YIELD.items():
        crop_mask = (yields["crop"] == crop) & (yields["yield_bu_acre"] >= min_y)
        mask = mask | crop_mask
    yields = yields[mask].copy()

    yields = (
        yields.sort_values("production", ascending=False)
        .groupby(["fips", "year", "crop"])
        .first()
        .reset_index()
    )
    return yields


def detrend_yields(yields):
    """Remove quadratic time trend from yields within each county-crop.

    Args:
        yields: DataFrame with fips, year, crop, yield_bu_acre.

    Returns:
        DataFrame with yield_trend, yield_detrended added.
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
    """Construct county-year panel with treatment (farm income) and instrument (weather shock).

    Treatment (D): Farm income deviation from county mean.
        D_it = (income_it - income_i_bar) / income_i_bar

    Instrument (Z): Weather-driven income shock with fixed acreage weights.
        Z_it = sum_c[ yield_detrended_ict * acres_ic_bar * price_c ] / baseline_income_i

    Args:
        yields: DataFrame with detrended yields.
        cpi: DataFrame with year, cpi columns.

    Returns:
        County-year DataFrame with farm_income_deviation, weather_income_shock.
    """
    yields = yields.merge(cpi[["year", "cpi"]], on="year", how="left")
    deflator = CPI_2023 / yields["cpi"]
    yields["price"] = yields["crop"].map(COMMODITY_PRICES)

    yields["revenue_2023usd"] = (
        yields["yield_bu_acre"] * yields["acres_harvested"] * yields["price"] * deflator
    )

    mean_acres = (
        yields.groupby(["fips", "crop"])["acres_harvested"]
        .mean()
        .rename("acres_mean")
    )
    yields = yields.merge(mean_acres, on=["fips", "crop"], how="left")

    yields["weather_revenue_2023usd"] = (
        yields["yield_detrended"] * yields["acres_mean"] * yields["price"] * deflator
    )

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

    baseline = county_year.groupby("fips")["farm_income_proxy"].mean().rename("baseline_income")
    county_year = county_year.merge(baseline, on="fips", how="left")

    county_year["farm_income_deviation"] = (
        (county_year["farm_income_proxy"] - county_year["baseline_income"])
        / county_year["baseline_income"]
    )
    county_year["weather_income_shock"] = (
        county_year["weather_revenue"] / county_year["baseline_income"]
    )

    county_year = county_year[county_year["baseline_income"] > 1_000_000].copy()
    county_counts = county_year.groupby("fips").size()
    good_counties = county_counts[county_counts >= 10].index
    county_year = county_year[county_year["fips"].isin(good_counties)].copy()

    return county_year


def load_migration_outcomes():
    """Load ACS migration data and build all outcome variables.

    ACS B07001 column mislabeling fix (confirmed 2026-03-18):
        'moved_diff_county_same_state' = B07001_002E (same house, non-movers, ~87%)
        'moved_diff_state'             = B07001_049E (diff county same state in-movers, ~6%)
        'moved_from_abroad'            = B07001_065E (diff state in-movers, ~2%)

    Outcomes constructed:
        Spec A  : net_outmigration_rate = -(pop_t - pop_{t-1}) / pop_{t-1}
        Spec A2 : same, but filtered to counties with |pop_change| <= 10% (no boundary changes)
                  and smoothed via 3-year rolling average
        Spec B  : gross_mobility_rate = (mobility_total - same_house) / total_population
                  = fraction of current residents who lived elsewhere 1yr ago (in-migration proxy)
        Spec C  : true_diff_county_in_rate = B07001_049E / total_population (PRIMARY)
        Spec D  : long_distance_in_rate = (diff_county + diff_state) / total_population
        Spec E  : gross_out_rate = net_outmigration_rate + true_diff_county_in_rate
                  = (-pop_change + in_migration) / baseline_pop ≈ out-migration rate

    Returns:
        DataFrame with fips, year, all outcome variables, total_population.
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

    # Rename to true content (correct the mislabeling)
    mig = mig.rename(columns={
        "moved_diff_county_same_state": "same_house",           # B07001_002E
        "moved_diff_state": "true_diff_county_same_state",     # B07001_049E
        "moved_from_abroad": "true_diff_state",                # B07001_065E
    })

    # ── Population-based outcomes ──
    demo = demo.sort_values(["fips", "year"]).copy()
    demo["pop_change_rate"] = demo.groupby("fips")["total_population"].pct_change()

    # Flag extreme population changes (likely boundary/reclassification events)
    demo["extreme_change"] = abs(demo["pop_change_rate"]) > POP_CHANGE_EXTREME

    # 3-year rolling average population change (annualised from 3yr window)
    demo["pop_change_3yr"] = (
        demo.groupby("fips")["total_population"]
        .pct_change(periods=3) / 3.0
    )

    # Baseline population (mean over sample) for weighting
    demo["baseline_pop"] = demo.groupby("fips")["total_population"].transform("mean")

    panel = mig.merge(demo, on=["fips", "year"], how="inner")

    # ── Spec A: raw net outmigration ──
    panel["outmigration_rate"] = -panel["pop_change_rate"]

    # ── Spec A2: cleaned net outmigration (no boundary changes) ──
    panel["outmigration_rate_clean"] = np.where(
        panel["extreme_change"],
        np.nan,
        -panel["pop_change_rate"],
    )

    # ── Spec A3: 3-year smoothed net outmigration (Approach 1) ──
    panel["outmigration_rate_3yr"] = -panel["pop_change_3yr"]

    # ── Spec B: gross mobility rate = in-movers / total pop (Approach 2) ──
    # = 1 - (same_house / total_pop) = movers in from anywhere / total
    panel["gross_mobility_rate"] = (
        (panel["mobility_total"] - panel["same_house"]).clip(lower=0)
        / panel["total_population"].replace(0, np.nan)
    )

    # ── Spec C: true inter-county in-migration (B07001_049E) ──
    panel["true_diff_county_in_rate"] = (
        panel["true_diff_county_same_state"].fillna(0)
        / panel["total_population"].replace(0, np.nan)
    )

    # ── Spec D: long-distance in-migration ──
    panel["long_distance_in_rate"] = (
        (panel["true_diff_county_same_state"].fillna(0)
         + panel["true_diff_state"].fillna(0))
        / panel["total_population"].replace(0, np.nan)
    )

    # ── Spec E: gross out-migration (Approach 3) ──
    # Identity: net_out = gross_out - gross_in
    # => gross_out = net_out + gross_in
    # gross_in ≈ true_diff_county_in_rate (in-movers / current pop)
    # We use raw pop change for net (less smoothing), same_house as in-migration proxy
    panel["gross_out_rate"] = (
        panel["outmigration_rate"].fillna(0) + panel["true_diff_county_in_rate"].fillna(0)
    )
    # Only valid when both components are present
    panel["gross_out_rate"] = np.where(
        panel["outmigration_rate"].isna() | panel["true_diff_county_in_rate"].isna(),
        np.nan,
        panel["gross_out_rate"],
    )

    # Filter to Corn Belt rural counties
    panel["state_fips"] = panel["fips"].str[:2]
    panel = panel[
        (panel["state_fips"].isin(CORN_BELT_STATE_FIPS))
        & (panel["total_population"] < POP_CAP)
        & (panel["total_population"] > 0)
    ].copy()

    return panel


# ──────────────────────────────────────────────────────────────────────
# 2SLS estimator
# ──────────────────────────────────────────────────────────────────────

def manual_2sls(panel, dep_var, endog_var, instrument_var,
                entity_col="fips", time_col="year",
                weight_col=None, label=""):
    """Estimate IV/2SLS with two-way FE via manual two-stage procedure.

    Stage 1: Regress demeaned endogenous variable on demeaned instrument.
    Stage 2: Regress demeaned outcome on demeaned fitted values.

    Optionally applies population weights (WLS) before demeaning for
    noise reduction in small counties.

    Args:
        panel: DataFrame with all variables.
        dep_var: Dependent variable name.
        endog_var: Endogenous treatment name.
        instrument_var: Instrument name.
        entity_col: Entity column (county FE).
        time_col: Time column (year FE).
        weight_col: Optional population weight column (sqrt applied internally).
        label: Label for printing.

    Returns:
        Dict with elasticity, CI, F-stat, p-value, diagnostics.
    """
    needed = [entity_col, time_col, dep_var, endog_var, instrument_var]
    if weight_col:
        needed.append(weight_col)

    df = panel[needed].dropna().copy()

    # Also drop rows where instrument or treatment are non-finite
    mask_finite = (
        np.isfinite(df[dep_var]) &
        np.isfinite(df[endog_var]) &
        np.isfinite(df[instrument_var])
    )
    df = df[mask_finite].copy()

    n_obs = len(df)
    n_counties = df[entity_col].nunique()
    n_years = df[time_col].nunique()

    print(f"  Sample: {n_obs} county-years, {n_counties} counties, {n_years} years")
    print(f"  Years: {df[time_col].min()}-{df[time_col].max()}")

    if n_obs < 100:
        print(f"  WARNING: only {n_obs} observations — skipping.")
        return None

    entity_ids = df[entity_col].values
    time_ids = df[time_col].values

    # Optional population weights (square root so variance is proportional to 1/pop)
    if weight_col and weight_col in df.columns:
        w = np.sqrt(df[weight_col].values.astype(np.float64))
        w = w / w.mean()  # normalise
    else:
        w = np.ones(n_obs)

    y_raw = df[dep_var].values * w
    d_raw = df[endog_var].values * w
    z_raw = df[instrument_var].values * w

    y_dm = demean_twoway(y_raw, entity_ids, time_ids)
    d_dm = demean_twoway(d_raw, entity_ids, time_ids)
    z_dm = demean_twoway(z_raw, entity_ids, time_ids)

    # ── FIRST STAGE ──
    Z_mat = z_dm.reshape(-1, 1)
    fs = ols_fit(d_dm, Z_mat)
    t_stat_z = fs["t_stats"][0]
    first_stage_F = t_stat_z ** 2
    d_hat = fs["fitted"]

    print(f"  First stage:")
