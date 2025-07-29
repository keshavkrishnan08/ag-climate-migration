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
