"""
Compute bootstrap confidence intervals for the four headline findings.

Bootstrap unit: county (with replacement across counties), 1000 iterations, seed=42.
All dollar values in 2023 USD (inherited from upstream parquet files).
Saves results to state/confidence_intervals.json.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RNG = np.random.default_rng(42)
N_BOOT = 1000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
STATE = PROJECT_ROOT / "state"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_stat(values: np.ndarray, stat_fn, n_boot: int = N_BOOT, rng=RNG):
    """Draw n_boot bootstrap samples and apply stat_fn to each.

    Args:
        values: 1-D array of per-unit values to resample.
        stat_fn: callable that reduces an array to a scalar.
        n_boot: number of bootstrap iterations.
        rng: numpy Generator instance.

    Returns:
        Tuple (mean, ci_lo, ci_hi) where ci_lo/hi are 2.5th/97.5th percentiles.

    Raises:
        ValueError: if values is empty.
    """
    if len(values) == 0:
        raise ValueError("bootstrap_stat received an empty array")

    boot_stats = np.array([
        stat_fn(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    return float(boot_stats.mean()), float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


def to_billions(x: float) -> float:
    """Convert raw dollar value to billions USD."""
    return round(x / 1e9, 4)


# ---------------------------------------------------------------------------
# 1. Stranded assets — DCF
# ---------------------------------------------------------------------------

def ci_stranded_dcf() -> dict:
    """Bootstrap CI for total stranded DCF value across counties under SSP2-4.5.

    Resamples counties (rows) with replacement. Statistic: sum of
    stranded_value_total across the sample (then scaled to full-population
    total by preserving the per-county distribution).

    Returns:
        dict with mean, ci_lo, ci_hi in billions USD.
    """
    path = RESULTS / "stranded_assets" / "stranded_national_SSP245.parquet"
    df = pd.read_parquet(path, columns=["fips", "stranded_value_total"])

    # One row per county; take the per-county total directly
    county_vals = df.groupby("fips")["stranded_value_total"].sum().values

    mean_b, lo_b, hi_b = bootstrap_stat(county_vals, np.sum)
    log.info("DCF stranded: mean=$%.1fB  95CI=[$%.1fB, $%.1fB]",
             to_billions(mean_b), to_billions(lo_b), to_billions(hi_b))
    return {
        "mean_B": to_billions(mean_b),
        "ci_lo_B": to_billions(lo_b),
        "ci_hi_B": to_billions(hi_b),
        "n_counties": int(len(county_vals)),
        "method": "bootstrap_sum_county_DCF_SSP245",
    }


# ---------------------------------------------------------------------------
# 2. Stranded assets — Hedonic (2050)
# ---------------------------------------------------------------------------

def ci_stranded_hedonic() -> dict:
    """Bootstrap CI for hedonic stranded value filtered to target_year == 2050.

    Args: none (reads file directly).
    Returns:
        dict with mean, ci_lo, ci_hi in billions USD.

    Raises:
        ValueError: if no 2050 rows found.
    """
    path = RESULTS / "stranded_assets" / "hedonic_stranded.parquet"
    df = pd.read_parquet(path, columns=["fips", "stranded_total", "target_year"])
    df2050 = df[df["target_year"] == 2050].copy()

    if df2050.empty:
        raise ValueError("No rows with target_year == 2050 in hedonic_stranded.parquet")

    county_vals = df2050.groupby("fips")["stranded_total"].sum().values

    mean_b, lo_b, hi_b = bootstrap_stat(county_vals, np.sum)
    log.info("Hedonic stranded 2050: mean=$%.1fB  95CI=[$%.1fB, $%.1fB]",
             to_billions(mean_b), to_billions(lo_b), to_billions(hi_b))
    return {
        "mean_B": to_billions(mean_b),
        "ci_lo_B": to_billions(lo_b),
        "ci_hi_B": to_billions(hi_b),
        "n_counties": int(len(county_vals)),
        "target_year": 2050,
        "method": "bootstrap_sum_county_hedonic_2050",
    }


# ---------------------------------------------------------------------------
# 3. Cascade counties — tipping before 2040
# ---------------------------------------------------------------------------

def ci_cascade() -> dict:
    """Bootstrap CI for count of counties tipping before 2040 under SSP2-4.5.

    Resamples counties; statistic is count of tipping_year <= 2040.

    Returns:
        dict with mean, ci_lo, ci_hi as county counts.
    """
    path = RESULTS / "cascade" / "tipping_points_SSP245.parquet"
    df = pd.read_parquet(path, columns=["fips", "tipping_year"])

    # One row per county; binary flag for tipping before 2040
    county_df = df.groupby("fips")["tipping_year"].min().reset_index()
    tip_flag = (county_df["tipping_year"] <= 2040).astype(float).values

    mean_c, lo_c, hi_c = bootstrap_stat(tip_flag, np.sum)
    log.info("Cascade <=2040: mean=%.0f  95CI=[%.0f, %.0f]",
             mean_c, lo_c, hi_c)
    return {
        "mean_counties": round(mean_c, 1),
        "ci_lo_counties": round(lo_c, 1),
        "ci_hi_counties": round(hi_c, 1),
        "n_counties_total": int(len(tip_flag)),
        "n_counties_tip_2040_raw": int((county_df["tipping_year"] <= 2040).sum()),
        "method": "bootstrap_count_tipping_year_le_2040_SSP245",
    }


# ---------------------------------------------------------------------------
# 4. Insurance mispricing
# ---------------------------------------------------------------------------

def ci_insurance() -> dict:
    """Bootstrap CI for total mispricing and cross-subsidy under SSP2-4.5.

    Resamples county × crop rows (the natural unit of observation for mispricing).
    Statistics:
      - total_mispricing: |sum of annual_cross_subsidy| across all rows
      - underpriced_total: sum where direction == 'underpriced' (cross_subsidy < 0)
      - overpriced_total: sum where direction == 'overpriced' (cross_subsidy > 0)
      - cross_subsidy: sum of absolute values in each direction (symmetric measure)

    Returns:
        dict with per-category mean, ci_lo, ci_hi in billions USD/yr.
    """
    path = RESULTS / "insurance" / "mispricing_SSP245.parquet"
    df = pd.read_parquet(path, columns=["fips", "crop", "annual_cross_subsidy", "direction"])

    # Resample at the county × crop level
