"""
Market Efficiency Test for Climate Risk Pricing in Farmland Values.

Tests whether farmland markets are already capitalizing projected climate risk.
If efficient, counties with larger projected warming (delta_tmax_july 2040)
should show LOWER recent land-value appreciation.

Regression:
    Δlog(land_value)_{2012-2022} = α + β₁·delta_tmax_july_2040
                                    + β₂·Δlog(income) + β₃·Δlog(pop)
                                    + state_FE + ε

    β₁ < 0 and significant → markets partially price climate risk (stranded
                              value overstated; reduce by degree of anticipation)
    β₁ ≈ 0 (not significant) → markets blind to climate → "stranded" framing holds
    β₁ > 0 and significant  → markets move against climate signal → even more stranding

Outputs: results/stranded_assets/market_efficiency_test.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "stranded_assets"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "market_efficiency_test.json"

LV_PATH = ROOT / "data" / "raw" / "nass" / "nass_land_values.parquet"
CP_PATH = ROOT / "data" / "projections" / "county_climate_projections.parquet"
ACS_PATH = ROOT / "data" / "raw" / "census" / "acs_county_demographics.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_land_value_change(
    path: Path,
    year_early: int = 2012,
    year_late: int = 2022,
) -> pd.DataFrame:
    """Compute county-level change in log farmland value per acre.

    Uses the two Census-of-Agriculture years that bracket ~2015 to ~2023.
    NASS land-value surveys are conducted every 5 years; 2012 and 2022 are
    the closest available years to the requested window.

    Args:
        path: Path to nass_land_values.parquet.
        year_early: Starting year for change (default 2012).
        year_late: Ending year for change (default 2022).

    Returns:
        DataFrame with columns [fips, dlog_land_value, lv_early, lv_late].

    Raises:
        FileNotFoundError: If the parquet file is missing.
        ValueError: If requested years are absent in the data.
    """
    lv = pd.read_parquet(path, columns=["fips", "year", "land_value_per_acre"])
    # Filter county-level rows only (exclude state aggregates)
    lv = lv[lv["fips"].str.len() == 5].copy()
    # Drop FIPS ending in 998/999 (NASS sub-state aggregates)
    lv = lv[~lv["fips"].str[-3:].isin(["998", "999"])]

    available_years = sorted(lv["year"].unique())
    if year_early not in available_years:
        raise ValueError(
            f"year_early={year_early} not in data; available: {available_years}"
        )
    if year_late not in available_years:
        raise ValueError(
            f"year_late={year_late} not in data; available: {available_years}"
        )

    early = (
        lv[lv["year"] == year_early]
        .set_index("fips")["land_value_per_acre"]
        .rename("lv_early")
    )
    late = (
        lv[lv["year"] == year_late]
        .set_index("fips")["land_value_per_acre"]
        .rename("lv_late")
    )

    df = pd.concat([early, late], axis=1).dropna()
    # Require strictly positive values to take logs
    df = df[(df["lv_early"] > 0) & (df["lv_late"] > 0)]
    df["dlog_land_value"] = np.log(df["lv_late"]) - np.log(df["lv_early"])
    df = df.reset_index().rename(columns={"index": "fips"})
    return df[["fips", "dlog_land_value", "lv_early", "lv_late"]]


def load_climate_warming(
    path: Path,
    scenario: str = "SSP245",
    year: int = 2040,
) -> pd.DataFrame:
    """Load projected July Tmax warming by county for a target year.

    Args:
        path: Path to county_climate_projections.parquet.
        scenario: Climate scenario string (default 'SSP245').
        year: Projection year for forward-looking warming signal (default 2040).

    Returns:
        DataFrame with columns [fips, delta_tmax_july_2040].

    Raises:
        FileNotFoundError: If the parquet file is missing.
    """
    cp = pd.read_parquet(
        path,
        columns=["fips", "year", "scenario", "delta_tmax_july"],
    )
    cp = cp[(cp["year"] == year) & (cp["scenario"] == scenario)].copy()
    cp = cp.rename(columns={"delta_tmax_july": f"delta_tmax_july_{year}"})
    return cp[["fips", f"delta_tmax_july_{year}"]].drop_duplicates("fips")


def load_acs_changes(
    path: Path,
    year_early: int = 2015,
    year_late: int = 2023,
) -> pd.DataFrame:
    """Compute county-level changes in log population and log income.

    Args:
        path: Path to acs_county_demographics.parquet.
        year_early: Starting ACS year (default 2015).
        year_late: Ending ACS year (default 2023).

    Returns:
        DataFrame with columns [fips, dlog_pop, dlog_income].

    Raises:
        FileNotFoundError: If the parquet file is missing.
        ValueError: If requested years are absent.
    """
    acs = pd.read_parquet(
        path,
        columns=["fips", "year", "total_population", "median_household_income"],
    )
    available = sorted(acs["year"].unique())
    if year_early not in available:
        raise ValueError(f"year_early={year_early} not in ACS; available: {available}")
    if year_late not in available:
        raise ValueError(f"year_late={year_late} not in ACS; available: {available}")

    def _slice(y: int) -> pd.DataFrame:
        return (
            acs[acs["year"] == y]
            .set_index("fips")[["total_population", "median_household_income"]]
            .rename(columns={
                "total_population": f"pop_{y}",
                "median_household_income": f"inc_{y}",
            })
        )

    early = _slice(year_early)
    late = _slice(year_late)
    df = pd.concat([early, late], axis=1).dropna()
    df = df[(df[f"pop_{year_early}"] > 0) & (df[f"pop_{year_late}"] > 0)]
    df = df[(df[f"inc_{year_early}"] > 0) & (df[f"inc_{year_late}"] > 0)]
    df["dlog_pop"] = np.log(df[f"pop_{year_late}"]) - np.log(df[f"pop_{year_early}"])
    df["dlog_income"] = np.log(df[f"inc_{year_late}"]) - np.log(df[f"inc_{year_early}"])
    return df.reset_index()[["fips", "dlog_pop", "dlog_income"]]


def build_state_fips(fips_series: pd.Series) -> pd.Series:
    """Extract 2-digit state FIPS from 5-digit county FIPS.

    Args:
        fips_series: Series of 5-digit county FIPS strings.

    Returns:
        Series of 2-digit state FIPS strings.
    """
    return fips_series.str[:2]


def interpret_result(beta: float, pvalue: float, alpha: float = 0.05) -> str:
    """Map regression coefficient and p-value to an economic interpretation.

    Args:
        beta: OLS coefficient on delta_tmax_july_2040.
        pvalue: Two-sided p-value for the coefficient.
        alpha: Significance threshold (default 0.05).

    Returns:
        String interpretation for the paper.
    """
    significant = pvalue < alpha
    if not significant:
        return (
            "Markets are NOT pricing climate risk (β₁ not significant). "
            "Land values have appreciated independently of projected warming. "
            "This validates the 'stranded asset' framing: current prices do not "
            "reflect forward-looking climate exposure."
        )
    if beta < 0:
        return (
            f"Markets are PARTIALLY pricing climate risk (β₁={beta:.4f}, "
            f"p={pvalue:.4f}). Counties with greater projected warming show "
            "lower appreciation. Stranded-value estimates should be reduced by "
            "the degree of market anticipation already embedded in prices."
        )
    # beta > 0 and significant
    return (
        f"Markets are moving AGAINST climate signals (β₁={beta:.4f}, "
        f"p={pvalue:.4f}). Counties facing more warming show HIGHER appreciation. "
        "This implies even greater stranding than our baseline estimates."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_market_efficiency_test() -> dict:
    """Run the full market efficiency test and return results as a dict.

    Returns:
        Dict with regression statistics, interpretation, and metadata.

    Raises:
        FileNotFoundError: If any required data file is missing.
    """
    print("Loading land values (2012 → 2022)...")
    lv_df = load_land_value_change(LV_PATH, year_early=2012, year_late=2022)
    print(f"  Land value counties: {len(lv_df):,}")

    print("Loading climate projections (SSP245, 2040)...")
    cp_df = load_climate_warming(CP_PATH, scenario="SSP245", year=2040)
    print(f"  Climate counties: {len(cp_df):,}")

    print("Loading ACS demographics (2015 → 2023)...")
    acs_df = load_acs_changes(ACS_PATH, year_early=2015, year_late=2023)
    print(f"  ACS counties: {len(acs_df):,}")

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    print("Merging datasets on FIPS...")
    df = (
        lv_df
        .merge(cp_df, on="fips", how="inner")
        .merge(acs_df, on="fips", how="inner")
