"""Robustness checks for Nature Food reviewers.

Six checks addressing the top reviewer concerns:
  1. Hedonic with soil quality proxy (historical yield 1990-2005 as soil proxy)
  2. Leave-one-crop-out sensitivity for stranded asset computation
  3. Leave-one-GCM-out sensitivity for stranded asset computation
  4. Placebo test: run cascade on LEAST climate-affected counties (top quartile positive impact)
  5. Temporal stability of the hedonic regression (2010-2015 vs 2015-2022)
  6. Insurance mispricing under alternative coverage levels (65% and 85%)

Each check saves results to results/robustness/ and prints a one-line verdict:
  ROBUST   — result is insensitive to the specification change
  SENSITIVE — result changes materially; report both
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from loguru import logger
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROJ    = PROJECT_ROOT / "data" / "projections"
RESULTS_DIR  = PROJECT_ROOT / "results" / "robustness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

CPI_2023 = 304.7
CPI_2022 = 296.8
DEFLATOR_2022 = CPI_2023 / CPI_2022
SEED = 42
np.random.seed(SEED)

GROWING_MONTHS = [4, 5, 6, 7, 8, 9]

COMMODITY_PRICES = {
    "corn": 5.50, "soybeans": 12.80, "wheat_winter": 7.20,
    "wheat_spring": 8.10, "cotton": 0.78, "sorghum": 5.30,
    "barley": 6.10, "oats": 3.80,
}

# From the main hedonic run — reference coefficients for stability comparison
BASELINE_BETA_TMAX     = None   # filled in at runtime from check 1 baseline run
BASELINE_STRANDED_B    = 168.0  # hedonic $168B from headline_numbers_preliminary.json
DCF_CONSERVATIVE_B     = 76.0   # DCF conservative from same file

VERDICTS: list[dict] = []


# ──────────────────────────────────────────────────────────────────────────────
# Shared data loader (load once, pass around)
# ──────────────────────────────────────────────────────────────────────────────

def load_shared_data() -> dict:
    """Load all datasets needed by robustness checks.

    Returns:
        Dict with keys: land_values, climate_monthly, acs, nass_yields,
        climate_proj, yield_proj, rma_data, tipping_df.
    """
    logger.info("Loading shared datasets…")

    land_values = pd.read_parquet(
        DATA_RAW / "nass" / "nass_land_values.parquet",
        columns=["fips", "year", "land_value_per_acre"],
    )
    climate_monthly = pd.read_parquet(
        DATA_RAW / "prism" / "county_climate_monthly.parquet",
        columns=(["fips", "year", "tmax_m07"] +
                 [f"precip_m{m:02d}" for m in GROWING_MONTHS]),
    )
    acs = pd.read_parquet(
        DATA_RAW / "census" / "acs_county_demographics.parquet",
        columns=["fips", "year", "total_population", "median_household_income"],
    )
    nass_yields = pd.read_parquet(
        DATA_RAW / "nass" / "nass_county_yields.parquet",
        columns=["fips", "year", "crop", "yield_bu_acre", "acres_harvested"],
    )
    climate_proj = pd.read_parquet(
        DATA_PROJ / "county_climate_projections.parquet",
        columns=["fips", "year", "scenario", "delta_tmax_july", "delta_precip_growing"],
    )
    climate_proj = climate_proj[climate_proj["scenario"] == "SSP245"].copy()

    yield_proj = pd.read_parquet(
        DATA_PROJ / "yield_projections_SSP245.parquet",
        columns=["fips", "year", "crop", "scenario",
                 "yield_projected", "yield_baseline",
                 "climate_impact_bu", "acres_harvested"],
    )

    rma_path = DATA_RAW / "rma" / "rma_sob_all_years.parquet"
    rma_data = pd.read_parquet(
        rma_path,
        columns=["year", "fips", "crop_name", "acres", "total_premium",
                 "indemnity", "premium_per_acre"],
    ) if rma_path.exists() else pd.DataFrame()

    tipping_df = pd.read_parquet(
        PROJECT_ROOT / "results" / "cascade" / "tipping_points_SSP245.parquet"
    )

    logger.info(
        f"  land_values={len(land_values)}, climate_monthly={len(climate_monthly)}, "
        f"nass_yields={len(nass_yields)}, yield_proj={len(yield_proj)}, "
        f"tipping_df={len(tipping_df)}"
    )
    return dict(
        land_values=land_values,
        climate_monthly=climate_monthly,
        acs=acs,
        nass_yields=nass_yields,
        climate_proj=climate_proj,
        yield_proj=yield_proj,
        rma_data=rma_data,
        tipping_df=tipping_df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Hedonic helper — reused across checks 1 and 5
# ──────────────────────────────────────────────────────────────────────────────

def _build_cross_section(
    land_values: pd.DataFrame,
    climate_monthly: pd.DataFrame,
    acs: pd.DataFrame,
    nass_yields: pd.DataFrame,
    lv_years: list[int],
    clim_years: tuple[int, int],
    acs_years: tuple[int, int],
) -> pd.DataFrame:
    """Build hedonic cross-section for a given temporal window.

    Args:
        land_values: NASS land value panel.
        climate_monthly: PRISM monthly climate panel.
        acs: ACS demographics panel.
        nass_yields: NASS county yields (for farm-acres calibration).
        lv_years: Census of Ag years to average for land value (e.g. [2017, 2022]).
        clim_years: (start, end) inclusive for climate window average.
        acs_years: (start, end) inclusive for ACS window average.

    Returns:
        Cross-section DataFrame with one row per county.
    """
    # Land value
    lv = land_values[land_values["year"].isin(lv_years)].copy()
    if 2022 in lv_years:
        lv.loc[lv["year"] == 2022, "land_value_per_acre"] *= DEFLATOR_2022
    lv_cs = lv.groupby("fips")["land_value_per_acre"].mean().reset_index()

    lo = np.percentile(lv_cs["land_value_per_acre"], 1)
    hi = np.percentile(lv_cs["land_value_per_acre"], 99)
    lv_cs = lv_cs[lv_cs["land_value_per_acre"].between(lo, hi)].copy()

    # Climate
    clim_w = climate_monthly[
        climate_monthly["year"].between(*clim_years)
    ].copy()
    precip_cols = [f"precip_m{m:02d}" for m in GROWING_MONTHS]
    clim_w["precip_growing"] = clim_w[precip_cols].sum(axis=1)
    clim_w["tmax_july"] = clim_w["tmax_m07"]
    clim_cs = (
        clim_w.groupby("fips")[["tmax_july", "precip_growing"]]
        .mean().reset_index()
    )

    # ACS
    acs_w = acs[acs["year"].between(*acs_years)].copy()
    acs_cs = (
        acs_w.groupby("fips")[["total_population", "median_household_income"]]
        .mean().reset_index()
    )

    # Farm acres (use 2012-2022 NASS as best available)
    ny = nass_yields[nass_yields["year"].between(2012, 2022)].copy()
    max_by = ny.groupby(["fips", "year"])["acres_harvested"].max()
    fa = max_by.groupby("fips").mean().reset_index().rename(
        columns={"acres_harvested": "farm_acres"}
    )

    df = (
        lv_cs
        .merge(clim_cs, on="fips", how="inner")
        .merge(acs_cs,  on="fips", how="inner")
        .merge(fa,      on="fips", how="left")
    )
    df["tmax_july_sq"]   = df["tmax_july"] ** 2
    df["log_land_value"] = np.log(df["land_value_per_acre"])
    df["log_pop"]        = np.log(df["total_population"].clip(lower=1))
    df["log_income"]     = np.log(df["median_household_income"].clip(lower=1))
    df["state_fips"]     = df["fips"].str[:2]
    df = df.dropna(subset=["log_land_value", "tmax_july", "precip_growing",
                            "log_pop", "log_income"])
    df = df[(df["total_population"] > 0) &
            (df["median_household_income"] > 0) &
            (df["tmax_july"] > 30) &
            (df["precip_growing"] >= 0)]
    return df


def _run_hedonic_ols(df: pd.DataFrame, extra_vars: str = "") -> smf.ols:
    """Fit hedonic OLS with optional extra RHS variables.

    Args:
        df: Cross-section DataFrame from _build_cross_section.
        extra_vars: Additional formula terms, e.g. '+ log_yield_baseline'.

    Returns:
        Fitted OLS results wrapper.
    """
    formula = (
        "log_land_value ~ tmax_july + tmax_july_sq + precip_growing "
        f"+ log_pop + log_income {extra_vars} + C(state_fips)"
    )
    return smf.ols(formula=formula, data=df).fit(cov_type="HC3")


def _hedonic_stranded(
    df: pd.DataFrame,
    result,
    climate_proj: pd.DataFrame,
    target_year: int = 2050,
) -> float:
    """Compute total stranded value ($B) from a fitted hedonic and projections.

    Args:
        df: Cross-section DataFrame (must include farm_acres, land_value_per_acre,
            tmax_july, tmax_july_sq, precip_growing).
        result: Fitted OLS results.
        climate_proj: Projection DataFrame (fips, year, delta_tmax_july,
            delta_precip_growing).
        target_year: Projection year.

    Returns:
        Total stranded value in $B (losses only, counties where warming hurts).
    """
    proj_yr = climate_proj[climate_proj["year"] == target_year].copy()
    df2 = df.merge(
        proj_yr[["fips", "delta_tmax_july", "delta_precip_growing"]],
        on="fips", how="inner",
    )

    b_T  = result.params.get("tmax_july",    0)
    b_T2 = result.params.get("tmax_july_sq", 0)
    b_P  = result.params.get("precip_growing", 0)

    df2["climate_hat_curr"] = (
        b_T  * df2["tmax_july"] +
        b_T2 * df2["tmax_july_sq"] +
        b_P  * df2["precip_growing"]
    )
    tmax_proj = df2["tmax_july"] + df2["delta_tmax_july"]
    df2["climate_hat_proj"] = (
        b_T  * tmax_proj +
        b_T2 * tmax_proj**2 +
        b_P  * df2["precip_growing"]   # precip held constant (conservative)
    )
    df2["delta_log_lv"] = df2["climate_hat_curr"] - df2["climate_hat_proj"]
    df2["delta_lv_per_acre"] = (
        df2["land_value_per_acre"] * (1 - np.exp(-df2["delta_log_lv"]))
    )
    df2["farm_acres"] = df2["farm_acres"].fillna(0)
    df2["stranded"] = df2["delta_lv_per_acre"] * df2["farm_acres"]
    return float(df2[df2["stranded"] > 0]["stranded"].sum() / 1e9)


# ──────────────────────────────────────────────────────────────────────────────
# Insurance helper — reused for check 6
# ──────────────────────────────────────────────────────────────────────────────

RMA_CROP_MAP = {
    "CORN": "corn", "SOYBEANS": "soybeans", "WHEAT": "wheat_winter",
    "COTTON": "cotton", "GRAIN SORGHUM": "sorghum",
    "BARLEY": "barley", "OATS": "oats",
}

CROP_MIN_APH = {
    "corn": 50.0, "soybeans": 10.0, "wheat_winter": 5.0,
    "cotton": 100.0, "sorghum": 10.0, "barley": 10.0, "oats": 5.0,
}


def _expected_indemnity(K: float, mu: float, sigma: float) -> float:
    """Analytical put formula: E[max(K - X, 0)] where X ~ N(mu, sigma²).

    Args:
        K: Revenue guarantee level ($/acre).
        mu: Expected revenue ($/acre).
        sigma: Revenue standard deviation ($/acre).

    Returns:
        Expected indemnity per acre ($).
    """
    sigma = max(sigma, 1.0)
    z = (K - mu) / sigma
    return float(max((K - mu) * scipy_stats.norm.cdf(z) +
                     sigma * scipy_stats.norm.pdf(z), 0.0))


def _compute_insurance_at_coverage(
    rma_data: pd.DataFrame,
    yield_proj: pd.DataFrame,
    nass_yields: pd.DataFrame,
    coverage: float = 0.75,
) -> dict:
    """Compute national insurance mispricing at a given coverage level.

    Args:
        rma_data: RMA Summary of Business raw data.
        yield_proj: Yield projections DataFrame.
        nass_yields: NASS historical yields for CV computation.
        coverage: Coverage level (e.g. 0.65, 0.75, 0.85).

    Returns:
        Dict with total_mispricing_B, cross_subsidy_B, underpriced_B, overpriced_B.
    """
    MAX_EI_RATIO = 5.0
    FUTURE_START, FUTURE_END = 2040, 2050

    # RMA aggregation
    rma = rma_data.copy()
    rma["crop_name_clean"] = rma["crop_name"].str.strip().str.upper()
    rma["crop"] = rma["crop_name_clean"].map(RMA_CROP_MAP)
    rma = rma[rma["crop"].notna()].copy()
    rma_recent = rma[rma["year"] >= rma["year"].max() - 10]
    rma_by_yr = (
        rma_recent
        .groupby(["fips", "crop", "year"], as_index=False)
        .agg(acres_yr=("acres", "sum"),
             premium_total_yr=("total_premium", "sum"))
    )
    rma_agg = (
        rma_by_yr
        .groupby(["fips", "crop"], as_index=False)
        .agg(insured_acres=("acres_yr", "mean"),
             total_premium=("premium_total_yr", "mean"))
    )
    rma_agg["premium_per_acre"] = (
        rma_agg["total_premium"] / rma_agg["insured_acres"].replace(0, np.nan)
    )
    rma_agg = rma_agg[rma_agg["insured_acres"] > 0].copy()

    # Yield CV from NASS
    nass_recent = nass_yields[
        nass_yields["year"].between(2008, 2023) & (nass_yields["yield_bu_acre"] > 0)
    ]
    cv_df = (
        nass_recent
        .groupby(["fips", "crop"])
        .agg(hist_mean=("yield_bu_acre", "mean"),
             hist_std=("yield_bu_acre", "std"),
             n_obs=("yield_bu_acre", "count"))
        .reset_index()
    )
    cv_df = cv_df[cv_df["n_obs"] >= 5].copy()
    cv_df["yield_cv"] = (cv_df["hist_std"] / cv_df["hist_mean"]).clip(0.05, 0.50).fillna(0.20)
    crop_med_cv = cv_df.groupby("crop")["yield_cv"].median().to_dict()

    # APH and future yields
    aph = yield_proj.groupby(["fips", "crop"], as_index=False).agg(
        aph_yield=("yield_baseline", "mean")
    )
    future = (
        yield_proj[yield_proj["year"].between(FUTURE_START, FUTURE_END)]
        .groupby(["fips", "crop"], as_index=False)
        .agg(future_yield=("yield_projected", "mean"))
    )
    proj = aph.merge(future, on=["fips", "crop"], how="inner")
    proj["future_yield"] = proj["future_yield"].clip(lower=0.0)

    # Filter fringe counties
    mask = pd.Series(True, index=proj.index)
    for crop, thresh in CROP_MIN_APH.items():
        mask &= ~((proj["crop"] == crop) & (proj["aph_yield"] < thresh))
    proj = proj[mask & (proj["aph_yield"] > 0)].copy()

    # Attach CV
    proj = proj.merge(cv_df[["fips", "crop", "yield_cv"]], on=["fips", "crop"], how="left")
    for crop in proj["crop"].unique():
        fill_mask = (proj["crop"] == crop) & proj["yield_cv"].isna()
        proj.loc[fill_mask, "yield_cv"] = crop_med_cv.get(crop, 0.20)
    proj["yield_cv"] = proj["yield_cv"].fillna(0.20).clip(0.05, 0.50)

    # EI ratio at requested coverage level
    proj["price"] = proj["crop"].map(COMMODITY_PRICES).fillna(5.0)
    proj["K"]         = proj["aph_yield"] * coverage * proj["price"]
    proj["sigma_rev"] = proj["aph_yield"] * proj["yield_cv"] * proj["price"]
    proj["mu_future"] = proj["future_yield"] * proj["price"]
    proj["mu_aph"]    = proj["aph_yield"]    * proj["price"]

    proj["ei_future"] = proj.apply(
        lambda r: _expected_indemnity(r["K"], r["mu_future"], r["sigma_rev"]), axis=1
    )
    proj["ei_aph"] = proj.apply(
        lambda r: _expected_indemnity(r["K"], r["mu_aph"], r["sigma_rev"]), axis=1
    )
    proj["ei_ratio"] = (
        proj["ei_future"] / proj["ei_aph"].replace(0, np.nan)
    ).clip(0.0, MAX_EI_RATIO).fillna(1.0)

    proj["yield_delta"]  = proj["future_yield"] - proj["aph_yield"]
    proj["direction"] = np.where(proj["yield_delta"] < 0, "underpriced",
                         np.where(proj["yield_delta"] > 0, "overpriced", "fair"))

    df = proj.merge(
        rma_agg[["fips", "crop", "insured_acres", "premium_per_acre"]],
        on=["fips", "crop"], how="left"
    )
    df["insured_acres"]    = df["insured_acres"].fillna(0.0)
    df["premium_per_acre"] = df["premium_per_acre"].fillna(0.0)
    df["mispricing_per_acre"] = df["premium_per_acre"] * (df["ei_ratio"] - 1.0)
    df["annual_cross_subsidy"] = df["mispricing_per_acre"] * df["insured_acres"]

    df_rma = df[df["insured_acres"] > 0].copy()
    under_B = df_rma.loc[df_rma["direction"] == "underpriced", "annual_cross_subsidy"].sum() / 1e9
    over_B  = df_rma.loc[df_rma["direction"] == "overpriced",  "annual_cross_subsidy"].abs().sum() / 1e9
    xsub_B  = min(under_B, over_B)
    total_B = under_B + over_B

    return dict(
        coverage=coverage,
        underpriced_B=round(under_B, 2),
        overpriced_B=round(over_B, 2),
        cross_subsidy_B=round(xsub_B, 2),
        total_mispricing_B=round(total_B, 2),
    )


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 1: Hedonic with soil quality proxy
# ──────────────────────────────────────────────────────────────────────────────

def check1_hedonic_soil_proxy(data: dict) -> dict:
    """Add yield_baseline_proxy (mean yield 1990-2005) as soil quality control.

    Rationale: Counties with high historical yields have better soil (drainage,
    organic matter, pH). Without a soil control, climate coefficients could be
    capturing soil quality variation correlated with climate. If coefficients
    are stable with and without this proxy, the hedonic is robust.

    Args:
        data: Shared data dict from load_shared_data().

    Returns:
        Result dict with coefficients and stranded values for both specs.
    """
    logger.info("=" * 60)
    logger.info("CHECK 1: Hedonic with soil quality proxy (yield_baseline 1990-2005)")
    logger.info("=" * 60)

    lv = data["land_values"]
    cm = data["climate_monthly"]
    acs = data["acs"]
    ny = data["nass_yields"]
    cp = data["climate_proj"]

    # Build baseline cross-section (same as main hedonic)
    df_base = _build_cross_section(
        lv, cm, acs, ny,
        lv_years=[2017, 2022],
        clim_years=(2019, 2023),
        acs_years=(2019, 2023),
    )

    # Compute soil proxy: mean yield 1990-2005 per county, across all crops, weighted by acres
    early = ny[ny["year"].between(1990, 2005) & (ny["yield_bu_acre"] > 0)].copy()
    # Standardize yields within crop (different unit scales across crops)
    crop_stats = early.groupby("crop")["yield_bu_acre"].agg(["mean", "std"]).reset_index()
    crop_stats.columns = ["crop", "crop_mean", "crop_std"]
    early = early.merge(crop_stats, on="crop", how="left")
    early["yield_z"] = (early["yield_bu_acre"] - early["crop_mean"]) / early["crop_std"].replace(0, 1)

    # County mean standardized yield (soil signal), weighted by acres
    soil_proxy = (
        early.groupby("fips")
        .apply(lambda g: np.average(g["yield_z"], weights=g["acres_harvested"].clip(lower=0.01)))
        .reset_index()
        .rename(columns={0: "yield_baseline_z"})
    )

    logger.info(f"  Soil proxy computed for {len(soil_proxy)} counties")

    # Merge into cross-section
    df_soil = df_base.merge(soil_proxy, on="fips", how="inner")
    df_soil["log_yield_baseline"] = df_soil["yield_baseline_z"]  # already z-scored
    logger.info(f"  Matched counties (base + soil proxy): {len(df_soil)}, base only: {len(df_base)}")

    # Run baseline (without soil proxy)
    res_base = _run_hedonic_ols(df_base)
    # Run with soil proxy
    res_soil = _run_hedonic_ols(df_soil, extra_vars="+ log_yield_baseline")

    # Extract key coefficients
    vars_of_interest = ["tmax_july", "tmax_july_sq", "precip_growing", "log_pop", "log_income"]
    coef_comparison = {}
    for v in vars_of_interest:
        b0 = res_base.params.get(v, np.nan)
        b1 = res_soil.params.get(v, np.nan)
        pct_chg = abs(b1 - b0) / max(abs(b0), 1e-10) * 100
        coef_comparison[v] = {
            "baseline":    round(b0, 6),
            "with_soil":   round(b1, 6),
            "pct_change":  round(pct_chg, 1),
            "p_baseline":  round(res_base.pvalues.get(v, np.nan), 4),
            "p_with_soil": round(res_soil.pvalues.get(v, np.nan), 4),
        }

    soil_coef = res_soil.params.get("log_yield_baseline", np.nan)
    soil_p    = res_soil.pvalues.get("log_yield_baseline", np.nan)
    logger.info(f"  Soil proxy coef: {soil_coef:+.4f} (p={soil_p:.4f})")

    # Stranded value under both specs (2050 target)
    stranded_base = _hedonic_stranded(df_base, res_base, cp, target_year=2050)
    stranded_soil = _hedonic_stranded(df_soil, res_soil, cp, target_year=2050)

    # Stability threshold: tmax_july coefficient changes <20% → ROBUST
    tmax_chg = coef_comparison["tmax_july"]["pct_change"]
    stranded_chg_pct = abs(stranded_soil - stranded_base) / max(stranded_base, 0.1) * 100

    verdict = "ROBUST" if (tmax_chg < 20 and stranded_chg_pct < 20) else "SENSITIVE"

    summary = (
        f"{verdict} | Soil proxy: tmax_july coef changes {tmax_chg:.1f}% "
        f"(baseline={res_base.params.get('tmax_july', np.nan):.5f}, "
        f"w/soil={res_soil.params.get('tmax_july', np.nan):.5f}); "
        f"stranded ${stranded_base:.1f}B → ${stranded_soil:.1f}B "
        f"({stranded_chg_pct:.1f}% change); "
        f"soil proxy p={soil_p:.4f}"
    )
    logger.info(f"  VERDICT: {summary}")

    result = {
        "check": "hedonic_soil_proxy",
        "verdict": verdict,
        "summary": summary,
        "coefficients": coef_comparison,
        "soil_proxy_coef": float(soil_coef),
        "soil_proxy_p":    float(soil_p),
        "stranded_baseline_B": float(stranded_base),
        "stranded_with_soil_B": float(stranded_soil),
        "stranded_change_pct": float(stranded_chg_pct),
        "r2_baseline":  float(res_base.rsquared),
        "r2_with_soil": float(res_soil.rsquared),
        "n_baseline":   int(res_base.nobs),
        "n_with_soil":  int(res_soil.nobs),
    }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 2: Leave-one-crop-out sensitivity
# ──────────────────────────────────────────────────────────────────────────────

def check2_leave_one_crop_out(data: dict) -> dict:
    """Drop each crop and recompute DCF stranded asset.

    Uses yield_proj climate_impact_bu × acres × price as a proxy for each
    crop's contribution to total stranded value. The DCF conservative estimate
    of $76B is recomputed after dropping each crop from the income stream.

    Args:
        data: Shared data dict.

    Returns:
        Result dict with per-crop stranded values and verdict.
    """
    logger.info("=" * 60)
    logger.info("CHECK 2: Leave-one-crop-out sensitivity on stranded assets")
    logger.info("=" * 60)

    yp = data["yield_proj"]

    # Climate damage per county-crop-year: negative climate_impact_bu = yield loss
    # Total damage per crop = sum(|loss| × acres × price) for counties with negative impact
    future = yp[yp["year"].between(2040, 2050)].copy()
    future["price"] = future["crop"].map(COMMODITY_PRICES).fillna(5.0)
    future["climate_damage"] = (
