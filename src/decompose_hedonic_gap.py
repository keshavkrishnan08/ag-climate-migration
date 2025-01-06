"""SI Section 6: Decomposition of the Hedonic-DCF Gap ($168B vs $105B = $63B).

The hedonic regression captures ALL channels that affect farmland value.
The DCF captures only field-crop income. This script decomposes the $63B gap
into four economic channels:

    (a) Livestock/dairy heat stress   (~$20-25B)
    (b) Water availability            (~$15-20B)
    (c) Amenity/quality-of-life       (~$10-15B)
    (d) Specialty crops               (~$5-10B)

Method:
    1. Load hedonic (2050, SSP245) and DCF central (SR, r=3%, h=35) results.
    2. Merge on FIPS; compute per-county gap = hedonic_stranded - dcf_stranded.
    3. Correlate the gap with proxy indicators for each channel.
    4. Apportion $63B gap using regression coefficients as weights.
    5. Write results to results/decomposition/hedonic_dcf_decomposition.json
       and a LaTeX table fragment to paper/si_section6_decomposition.tex.

Args:
    None (reads from canonical paths).
Returns:
    Dict with decomposition results.
Raises:
    FileNotFoundError if input parquets are missing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HEDONIC_PARQUET = PROJECT_ROOT / "results/stranded_assets/hedonic_stranded_2050.parquet"
DCF_PARQUET = PROJECT_ROOT / "results/stranded_assets/stranded_national_SR_SSP245.parquet"
CLIMATE_PROJ = PROJECT_ROOT / "data/projections/county_climate_projections.parquet"
LAND_VALUES = PROJECT_ROOT / "data/raw/nass/nass_land_values.parquet"
CASH_RENT = PROJECT_ROOT / "data/raw/nass/nass_cash_rent.parquet"
RMA_PARQUET = PROJECT_ROOT / "data/raw/rma/rma_sob_all_years.parquet"
ACS_DEMO = PROJECT_ROOT / "data/raw/census/acs_county_demographics.parquet"

OUTPUT_DIR = PROJECT_ROOT / "results/decomposition"
