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
