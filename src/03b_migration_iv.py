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

    # Filter to Corn Belt rural counties
    panel["state_fips"] = panel["fips"].str[:2]
    panel = panel[
        (panel["state_fips"].isin(CORN_BELT_STATE_FIPS))
        & (panel["total_population"] < POP_CAP)
        & (panel["total_population"] > 0)
    ].copy()

    return panel


def demean_twoway(arr, entity_ids, time_ids):
    """Apply two-way demeaning (Frisch-Waugh-Lovell within transformation).

    For variable x: x_tilde = x - x_bar_i - x_bar_t + x_bar

    Args:
        arr: 1D array of values.
        entity_ids: Array of entity identifiers.
        time_ids: Array of time identifiers.

    Returns:
        Demeaned array.
    """
    arr = np.asarray(arr, dtype=np.float64)
    grand_mean = arr.mean()

    entity_uniq, entity_inv = np.unique(entity_ids, return_inverse=True)
    entity_sums = np.bincount(entity_inv, weights=arr)
    entity_counts = np.bincount(entity_inv)
    entity_means = entity_sums / entity_counts
    entity_mean_expanded = entity_means[entity_inv]

    time_uniq, time_inv = np.unique(time_ids, return_inverse=True)
    time_sums = np.bincount(time_inv, weights=arr)
    time_counts = np.bincount(time_inv)
    time_means = time_sums / time_counts
    time_mean_expanded = time_means[time_inv]

    return arr - entity_mean_expanded - time_mean_expanded + grand_mean


def manual_2sls(panel, dep_var, endog_var, instrument_var,
                entity_col="fips", time_col="year", label=""):
    """Estimate IV/2SLS with two-way fixed effects via manual two-stage procedure.

    Stage 1: D_tilde on Z_tilde (demeaned for FE)
    Stage 2: Y_tilde on D_hat_tilde

    Uses Frisch-Waugh-Lovell: demean all variables by county and year,
    then run simple OLS. This absorbs the fixed effects.

    Args:
        panel: DataFrame with all variables.
        dep_var: Name of dependent variable.
        endog_var: Name of endogenous treatment variable.
        instrument_var: Name of instrument.
        entity_col: Column for entity FE.
        time_col: Column for time FE.
        label: Label for printing.

    Returns:
        Dict with elasticity, CI, F-stat, diagnostics.
    """
    df = panel[[entity_col, time_col, dep_var, endog_var, instrument_var]].dropna().copy()
    n_obs = len(df)
    n_counties = df[entity_col].nunique()
    n_years = df[time_col].nunique()

    print(f"  Sample: {n_obs} county-years, {n_counties} counties, {n_years} years")
    print(f"  Years: {df[time_col].min()}-{df[time_col].max()}")

    entity_ids = df[entity_col].values
    time_ids = df[time_col].values

    # Two-way demeaning
    y_dm = demean_twoway(df[dep_var].values, entity_ids, time_ids)
    d_dm = demean_twoway(df[endog_var].values, entity_ids, time_ids)
    z_dm = demean_twoway(df[instrument_var].values, entity_ids, time_ids)

    # ── FIRST STAGE: D on Z (demeaned) ──
    Z_mat = z_dm.reshape(-1, 1)
    fs = ols_fit(d_dm, Z_mat)

    t_stat_z = fs["t_stats"][0]
    first_stage_F = t_stat_z ** 2

    print(f"  First stage:")
    print(f"    Coefficient on instrument: {fs['beta'][0]:.6f}")
    print(f"    t-statistic: {t_stat_z:.2f}")
    print(f"    F-statistic: {first_stage_F:.2f}")
    print(f"    R-squared (partial): {fs['r_squared']:.4f}")

    d_hat = fs["fitted"]

    # ── SECOND STAGE: Y on D_hat ──
    D_hat_mat = d_hat.reshape(-1, 1)
    ss = ols_fit(y_dm, D_hat_mat)
    beta_iv = ss["beta"][0]

    # ── Correct standard errors using actual endogenous residuals ──
    residuals = y_dm - beta_iv * d_dm
    dof = n_obs - n_counties - n_years + 1
    sigma2 = (residuals @ residuals) / max(dof, 1)
    d_hat_ss = d_hat @ d_hat
    var_beta = sigma2 / d_hat_ss
    se_beta = np.sqrt(var_beta)

    # ── Cluster-robust standard errors (county level) ──
    unique_clusters = np.unique(entity_ids)
    n_clusters = len(unique_clusters)
    scores = residuals * d_hat

    meat = 0.0
    for c in unique_clusters:
        mask = entity_ids == c
        s_c = scores[mask].sum()
        meat += s_c ** 2

    dof_correction = n_clusters / (n_clusters - 1)
    bread = 1.0 / d_hat_ss
    var_clustered = bread ** 2 * meat * dof_correction
    se_clustered = np.sqrt(var_clustered)

    se_final = max(se_beta, se_clustered)

    t_crit = stats.t.ppf(0.975, df=n_clusters - 1)
    ci_lower = beta_iv - t_crit * se_final
    ci_upper = beta_iv + t_crit * se_final

    iv_t = beta_iv / se_final
    iv_p = 2 * stats.t.sf(abs(iv_t), n_clusters - 1)

    print(f"\n  Second stage (IV/2SLS):")
    print(f"    Elasticity (beta_IV): {beta_iv:.6f}")
    print(f"    SE (homoskedastic):   {se_beta:.6f}")
    print(f"    SE (cluster-robust):  {se_clustered:.6f}")
    print(f"    SE (used):            {se_final:.6f}")
    print(f"    95% CI:               [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"    t-stat:               {iv_t:.2f}")
    print(f"    p-value:              {iv_p:.4f}")

    # ── Reduced form: Y on Z directly ──
    rf = ols_fit(y_dm, Z_mat)
    beta_rf = rf["beta"][0]
    rf_resid = rf["residuals"]
    rf_scores = rf_resid * z_dm
    rf_meat = 0.0
    for c in unique_clusters:
        mask = entity_ids == c
        s_c = rf_scores[mask].sum()
        rf_meat += s_c ** 2
    z_ss = z_dm @ z_dm
    rf_var_cl = (1.0 / z_ss) ** 2 * rf_meat * dof_correction
    rf_se_cl = np.sqrt(rf_var_cl)
    rf_t = beta_rf / rf_se_cl
    rf_p = 2 * stats.t.sf(abs(rf_t), n_clusters - 1)

    print(f"\n  Reduced form (Y on Z directly):")
    print(f"    Coefficient: {beta_rf:.6f}")
    print(f"    SE (cluster): {rf_se_cl:.6f}")
    print(f"    t-stat: {rf_t:.2f}")
    print(f"    p-value: {rf_p:.4f}")

    # ── OLS comparison ──
    D_mat = d_dm.reshape(-1, 1)
    ols = ols_fit(y_dm, D_mat)
    beta_ols = ols["beta"][0]
    print(f"\n  OLS comparison:")
    print(f"    OLS coefficient: {beta_ols:.6f}")
    print(f"    IV coefficient:  {beta_iv:.6f}")
    if abs(beta_ols) > 1e-10:
        print(f"    Ratio (IV/OLS):  {beta_iv / beta_ols:.2f}")

    return {
        "elasticity": float(beta_iv),
        "se": float(se_final),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "first_stage_F": float(first_stage_F),
        "first_stage_coeff": float(fs["beta"][0]),
        "first_stage_t": float(t_stat_z),
        "first_stage_partial_r2": float(fs["r_squared"]),
        "reduced_form_coeff": float(beta_rf),
        "reduced_form_se": float(rf_se_cl),
        "reduced_form_t": float(rf_t),
        "reduced_form_p": float(rf_p),
        "n_obs": int(n_obs),
        "n_counties": int(n_counties),
        "n_years": int(n_years),
        "n_clusters": int(n_clusters),
        "ols_coefficient": float(beta_ols),
        "residual_variance": float(sigma2),
        "iv_t_stat": float(iv_t),
        "iv_p_value": float(iv_p),
    }


def main():
    """Run the full IV estimation pipeline."""
    print("=" * 70)
    print("Fix 4: IV Estimation -- Farm Income -> Outmigration Elasticity")
    print("=" * 70)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Step 1: Load and clean yields ──
    print("\n[1/5] Loading and cleaning NASS yield data...")
    yields = load_and_clean_yields()
    print(f"  Loaded {len(yields)} crop-county-year records")
    print(f"  Crops: {sorted(yields['crop'].unique())}")
    print(f"  Counties: {yields['fips'].nunique()}")
    print(f"  Years: {yields['year'].min()}-{yields['year'].max()}")

    # ── Step 2: Detrend yields ──
    print("\n[2/5] Detrending yields (quadratic county-crop trends)...")
    yields = detrend_yields(yields)
    print(f"  Detrended {len(yields)} records")
    print(f"  Mean yield_detrended: {yields['yield_detrended'].mean():.2f} (should be ~0)")
    print(f"  Std yield_detrended:  {yields['yield_detrended'].std():.2f}")

    # ── Step 3: Build panel with treatment and instrument ──
    print("\n[3/5] Building IV panel (treatment + instrument)...")
    cpi = pd.read_csv(PROJECT_ROOT / "data/raw/other/cpi_annual.csv")
    county_year = build_iv_panel(yields, cpi)
    print(f"  County-year records: {len(county_year)}")
    print(f"  Counties: {county_year['fips'].nunique()}")
    print(f"  Mean farm income proxy: ${county_year['farm_income_proxy'].mean():,.0f}")
    print(f"  Mean weather_income_shock: {county_year['weather_income_shock'].mean():.4f}")
    print(f"  Sd  weather_income_shock:  {county_year['weather_income_shock'].std():.4f}")

    # ── Step 4: Load migration and merge ──
    print("\n[4/5] Loading ACS migration data and building analysis panel...")
    migration = load_migration_outcome()
    print(f"  Migration records (rural Corn Belt): {len(migration)}")
    print(f"  ACS column correction applied: moved_diff_county_same_state -> true_same_house (B07001_002E)")
    print(f"  ACS column correction applied: moved_diff_state -> true_diff_county_same_state (B07001_049E)")
    print(f"  ACS column correction applied: moved_from_abroad -> true_diff_state (B07001_065E)")
    print(f"  Mean outmigration rate (pop change): {migration['outmigration_rate'].mean():.4f}")
    print(f"  Mean true_diff_county_in_rate:       {migration['true_diff_county_in_rate'].mean():.4f}")
    print(f"  Mean long_distance_in_rate:           {migration['long_distance_in_rate'].mean():.4f}")

    # Merge all outcome columns
    outcome_cols = ["fips", "year", "outmigration_rate", "intercounty_inmigration_rate",
                    "true_diff_county_in_rate", "long_distance_in_rate", "total_population"]
    panel = county_year.merge(
        migration[outcome_cols],
        on=["fips", "year"],
        how="inner",
    )

    # Filter
    panel = panel[
        (panel["year"].between(2010, 2023))
        & panel["farm_income_deviation"].notna()
        & panel["weather_income_shock"].notna()
        & panel["outmigration_rate"].notna()
        & np.isfinite(panel["farm_income_deviation"])
        & np.isfinite(panel["weather_income_shock"])
        & np.isfinite(panel["outmigration_rate"])
    ].copy()

    # Winsorize extremes for all outcomes and treatment/instrument
    winsorize_cols = [
        "farm_income_deviation", "weather_income_shock", "outmigration_rate",
        "true_diff_county_in_rate", "long_distance_in_rate", "intercounty_inmigration_rate",
    ]
    for col in winsorize_cols:
        if col in panel.columns:
            valid = panel[col].dropna()
            if len(valid) > 10:
                p01, p99 = valid.quantile([0.01, 0.99])
                panel[col] = panel[col].clip(p01, p99)

    print(f"  Merged panel: {len(panel)} observations")
    print(f"  Counties: {panel['fips'].nunique()}")
    print(f"  Years: {panel['year'].min()}-{panel['year'].max()}")

    print(f"\n  Descriptive statistics (post-winsorize):")
    print(f"    outmigration_rate (net pop change): mean={panel['outmigration_rate'].mean():.4f}, "
          f"sd={panel['outmigration_rate'].std():.4f}")
    print(f"    true_diff_county_in_rate (FIXED):   mean={panel['true_diff_county_in_rate'].mean():.4f}, "
          f"sd={panel['true_diff_county_in_rate'].std():.4f}")
    print(f"    long_distance_in_rate:              mean={panel['long_distance_in_rate'].mean():.4f}, "
          f"sd={panel['long_distance_in_rate'].std():.4f}")
    print(f"    farm_income_deviation (frac):       mean={panel['farm_income_deviation'].mean():.4f}, "
          f"sd={panel['farm_income_deviation'].std():.4f}")
    print(f"    weather_income_shock:               mean={panel['weather_income_shock'].mean():.4f}, "
          f"sd={panel['weather_income_shock'].std():.4f}")

    # ── Step 5: Run IV/2SLS specifications ──
    print("\n[5/5] Running IV/2SLS estimations...")

    # Spec A: Net outmigration (from pop change) — legacy, noisy
    print("\n" + "=" * 50)
    print("SPEC A: Net outmigration rate (pop change) [legacy, noisy]")
    print("=" * 50)
    iv_a = manual_2sls(
        panel,
        dep_var="outmigration_rate",
        endog_var="farm_income_deviation",
        instrument_var="weather_income_shock",
        entity_col="fips",
        time_col="year",
        label="outmigration",
    )

    # Spec B: All-movers in-migration rate (legacy — confounds same-county moves)
    panel_b = panel[panel["intercounty_inmigration_rate"].notna()].copy()

    print("\n" + "=" * 50)
    print("SPEC B: All-movers in-migration rate [legacy, includes same-county]")
    print("=" * 50)
    iv_b = manual_2sls(
        panel_b,
        dep_var="intercounty_inmigration_rate",
        endog_var="farm_income_deviation",
        instrument_var="weather_income_shock",
        entity_col="fips",
        time_col="year",
        label="all_movers",
    )

    # Spec C: TRUE inter-county in-migration (CORRECTED — B07001_049E)
    # This is the primary fix: uses actual different-county same-state in-movers.
    # Expected sign: POSITIVE beta (higher income -> more people move in from other counties).
    panel_c = panel[panel["true_diff_county_in_rate"].notna()
                    & np.isfinite(panel["true_diff_county_in_rate"])].copy()

    print("\n" + "=" * 50)
    print("SPEC C: True inter-county in-migration (B07001_049E) [PRIMARY FIX]")
    print("=" * 50)
    iv_c = manual_2sls(
        panel_c,
        dep_var="true_diff_county_in_rate",
        endog_var="farm_income_deviation",
        instrument_var="weather_income_shock",
        entity_col="fips",
        time_col="year",
        label="true_diff_county",
    )

    # Spec D: Long-distance in-migration (diff county + diff state)
    panel_d = panel[panel["long_distance_in_rate"].notna()
                    & np.isfinite(panel["long_distance_in_rate"])].copy()

    print("\n" + "=" * 50)
    print("SPEC D: Long-distance in-migration (diff county + diff state) [ROBUSTNESS]")
    print("=" * 50)
    iv_d = manual_2sls(
        panel_d,
        dep_var="long_distance_in_rate",
        endog_var="farm_income_deviation",
        instrument_var="weather_income_shock",
        entity_col="fips",
        time_col="year",
        label="long_distance",
    )

    # ── Choose primary estimate ──
    # Spec C is primary: true inter-county in-migration (B07001_049E), correctly labeled.
    # Spec A is secondary: population change (noisy, but directionally consistent).
    # Spec D is robustness: adds diff-state in-migrants.
    primary = iv_c
    mean_primary = panel_c["true_diff_county_in_rate"].mean()

    mean_outmig = panel["outmigration_rate"].mean()
    mean_inmig_b = panel_b["intercounty_inmigration_rate"].mean()
    mean_long = panel_d["long_distance_in_rate"].mean()

    # Semi-elasticity: beta_IV / mean(outcome)
    semi_elast_c = iv_c["elasticity"] / mean_primary if abs(mean_primary) > 1e-6 else float("nan")
    semi_elast_d = iv_d["elasticity"] / mean_long if abs(mean_long) > 1e-6 else float("nan")
    semi_elast_a = iv_a["elasticity"] / mean_outmig if abs(mean_outmig) > 1e-6 else float("nan")
    semi_elast_b = iv_b["elasticity"] / mean_inmig_b if abs(mean_inmig_b) > 1e-6 else float("nan")

    print("\n" + "=" * 70)
    print("ELASTICITY COMPARISON")
    print("=" * 70)
    print(f"  Spec C (true diff-county in-migration, PRIMARY FIX):")
    print(f"    beta_IV = {iv_c['elasticity']:.6f}")
    print(f"    p-value = {iv_c['iv_p_value']:.4f}")
    print(f"    Semi-elasticity = {semi_elast_c:.4f}")
    print(f"    Reduced form = {iv_c['reduced_form_coeff']:.6f} (p = {iv_c['reduced_form_p']:.4f})")
    print(f"    First-stage F = {iv_c['first_stage_F']:.1f}")
    print(f"  Spec D (long-distance in-migration, robustness):")
    print(f"    beta_IV = {iv_d['elasticity']:.6f}")
    print(f"    p-value = {iv_d['iv_p_value']:.4f}")
    print(f"    First-stage F = {iv_d['first_stage_F']:.1f}")
    print(f"  Spec A (outmigration, pop change, legacy):")
    print(f"    beta_IV = {iv_a['elasticity']:.6f}")
    print(f"    p-value = {iv_a['iv_p_value']:.4f}")
    print(f"    First-stage F = {iv_a['first_stage_F']:.1f}")
    print(f"  Spec B (all-movers in-migration, legacy):")
    print(f"    beta_IV = {iv_b['elasticity']:.6f}")
    print(f"    p-value = {iv_b['iv_p_value']:.4f}")
    print(f"    First-stage F = {iv_b['first_stage_F']:.1f}")
    print(f"  Feng et al. (2010): -0.17 (semi-elasticity, outmigration direction)")

    # ── Gate check ──
    print("\n" + "=" * 50)
    if primary["first_stage_F"] > 10:
        gate = "PASS"
        print(f"GATE: PASS (first-stage F = {primary['first_stage_F']:.1f} > 10)")
        elasticity = primary["elasticity"]
        ci_lower = primary["ci_95_lower"]
        ci_upper = primary["ci_95_upper"]
    else:
        gate = "CONDITIONAL"
        print(f"GATE: CONDITIONAL (first-stage F = {primary['first_stage_F']:.1f} < 10)")
        print("  Weak instrument detected. Using literature fallback (Feng et al. 2010).")
        elasticity = -0.17
        ci_lower = -0.25
        ci_upper = -0.09
        primary["estimated_elasticity_before_fallback"] = primary["elasticity"]

    # ── Save results ──
    economic_params = {
        "migration_elasticity": {
            "estimate": float(elasticity),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "first_stage_F": float(primary["first_stage_F"]),
            "n_obs": primary["n_obs"],
            "n_counties": primary["n_counties"],
            "n_years": primary["n_years"],
            "sample_period": f"{panel_c['year'].min()}-{panel_c['year'].max()}",
            "sample_definition": "Rural Corn Belt (pop < 50k), 11 states",
            "instrument": "weather_income_shock = yield_detrended x mean_acres x price / baseline_income",
            "treatment": "farm_income_deviation = (income - baseline) / baseline, crop revenue proxy (2023 USD)",
            "outcome": (
                "true_diff_county_in_rate = B07001_049E (diff county same state in-movers) "
                "/ total_population. ACS column was mislabeled as 'moved_diff_state'; "
                "corrected 2026-03-18."
            ),
            "fixed_effects": "county + year (two-way, absorbed via demeaning)",
            "standard_errors": "cluster-robust (county level)",
            "gate": gate,
            "ols_coefficient": primary["ols_coefficient"],
            "first_stage_partial_r2": primary["first_stage_partial_r2"],
            "reduced_form_coeff": primary["reduced_form_coeff"],
            "reduced_form_p": primary["reduced_form_p"],
            "iv_p_value": primary["iv_p_value"],
            "semi_elasticity_spec_c": float(semi_elast_c),
            "semi_elasticity_spec_d": float(semi_elast_d),
            "semi_elasticity_spec_a": float(semi_elast_a),
            "semi_elasticity_spec_b": float(semi_elast_b),
            "spec_a_outmig_elasticity": iv_a["elasticity"],
            "spec_a_outmig_pval": iv_a["iv_p_value"],
            "spec_b_all_movers_elasticity": iv_b["elasticity"],
            "spec_d_long_dist_elasticity": iv_d["elasticity"],
            "spec_d_long_dist_pval": iv_d["iv_p_value"],
            "acs_mislabel_fix": (
                "moved_diff_county_same_state = B07001_002E (same house, non-movers). "
                "moved_diff_state = B07001_049E (true diff-county same-state in-movers). "
                "moved_from_abroad = B07001_065E (true diff-state in-movers). "
                "Fix applied 2026-03-18."
            ),
            "timestamp": timestamp,
        }
    }

    output_path = PROJECT_ROOT / "state" / "economic_params.json"
    with open(output_path, "w") as f:
        json.dump(economic_params, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Save detailed diagnostics
    diag_dir = PROJECT_ROOT / "results" / f"{timestamp}"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_path = diag_dir / "iv_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(
            {
                "spec_c_true_diff_county": {
                    "outcome": "true_diff_county_in_rate (B07001_049E, PRIMARY FIX)",
                    "results": iv_c,
                    "mean_outcome": float(mean_primary),
                    "semi_elasticity": float(semi_elast_c),
                },
                "spec_d_long_distance": {
                    "outcome": "long_distance_in_rate (diff county + diff state)",
                    "results": iv_d,
                    "mean_outcome": float(mean_long),
                    "semi_elasticity": float(semi_elast_d),
                },
                "spec_a_outmigration": {
                    "outcome": "net_outmigration_rate (pop change, legacy)",
                    "results": iv_a,
                    "mean_outcome": float(mean_outmig),
