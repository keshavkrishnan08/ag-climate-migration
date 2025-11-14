"""Fix 5 — Validate crop switching model against 4 historical events.

Uses NASS county acreage data (1950+) and nClimDiv climate data to test
whether a simple climate-feature LightGBM model can reproduce observed
crop switching patterns. Four validation events from SWARM spec §A3:

1. Sorghum expansion in southern Plains (1950-1975) — POSITIVE
2. Cotton retreat from Missouri/Tennessee (1980-2010) — POSITIVE
3. Winter wheat boundary shift in Kansas (1990-2010) — POSITIVE
4. Soybean adoption in Corn Belt (1960-1980) — NEGATIVE test

All four must pass before the switching model can be trusted for projections.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NASS_PATH = PROJECT_ROOT / "data" / "raw" / "nass" / "nass_county_yields.parquet"
CLIMATE_PATH = PROJECT_ROOT / "data" / "raw" / "prism" / "county_climate_annual.parquet"
CLIMATE_MONTHLY_PATH = PROJECT_ROOT / "data" / "raw" / "prism" / "county_climate_monthly.parquet"
OUTPUT_PATH = PROJECT_ROOT / "state" / "validation" / "historical_switching.json"

RANDOM_SEED = 42

# Kansas county centroid latitudes (FIPS -> latitude in degrees N).
# Source: US Census Bureau TIGER/Gazetteer county centroids.
KANSAS_COUNTY_LAT = {
    "20001": 37.88, "20003": 37.77, "20005": 38.91, "20007": 37.25,
    "20009": 38.79, "20011": 38.48, "20013": 37.25, "20015": 37.80,
    "20017": 37.79, "20019": 37.27, "20021": 38.47, "20023": 37.56,
    "20025": 38.06, "20027": 37.56, "20029": 39.78, "20031": 37.24,
    "20033": 38.48, "20035": 37.25, "20037": 38.07, "20039": 39.05,
    "20041": 39.35, "20043": 38.68, "20045": 38.53, "20047": 38.68,
    "20049": 39.79, "20051": 39.35, "20053": 37.77, "20055": 38.26,
    "20057": 38.52, "20059": 37.24, "20061": 38.69, "20063": 38.07,
    "20065": 39.78, "20067": 38.07, "20069": 39.33, "20071": 37.56,
    "20073": 37.22, "20075": 37.86, "20077": 37.88, "20079": 37.80,
    "20081": 38.68, "20083": 37.56, "20085": 38.48, "20087": 38.92,
    "20089": 39.79, "20091": 38.49, "20093": 38.07, "20095": 38.26,
    "20097": 39.34, "20099": 38.50, "20101": 37.88, "20103": 37.58,
    "20105": 38.94, "20107": 38.06, "20109": 39.78, "20111": 39.06,
    "20113": 37.55, "20115": 37.86, "20117": 39.79, "20119": 37.26,
    "20121": 38.26, "20123": 39.05, "20125": 38.92, "20127": 38.92,
    "20129": 38.66, "20131": 37.26, "20133": 39.83, "20135": 38.92,
    "20137": 37.57, "20139": 39.05, "20141": 38.26, "20143": 39.04,
    "20145": 37.25, "20147": 38.06, "20149": 38.27, "20151": 37.24,
    "20153": 38.91, "20155": 39.35, "20157": 37.65, "20159": 37.26,
    "20161": 39.05, "20163": 39.34, "20165": 39.79, "20167": 38.86,
    "20169": 37.66, "20171": 39.06, "20173": 37.68, "20175": 37.25,
    "20177": 38.65, "20179": 37.55, "20181": 37.88, "20183": 39.04,
    "20185": 38.68, "20187": 37.80, "20189": 37.26, "20191": 37.56,
    "20193": 38.91, "20195": 37.80, "20197": 39.33, "20199": 38.06,
    "20201": 39.06, "20203": 39.79, "20205": 38.87, "20207": 38.68,
    "20209": 39.11,
}


# -----------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------

def load_nass_acreage(
    state_fips: list[str],
    crops: list[str] | None = None,
    year_min: int = 1950,
    year_max: int = 2025,
) -> pd.DataFrame:
    """Load deduplicated NASS county acreage data for target states/crops.

    Takes the MAX of acres_harvested per (fips, year, crop) to collapse
    the survey-item cross-product artifact in the raw NASS bulk download.

    Args:
        state_fips: 2-digit state FIPS strings to include.
        crops: Crop names to filter. None means all.
        year_min: Start year (inclusive).
        year_max: End year (inclusive).

    Returns:
        DataFrame with columns [fips, year, crop, acres_harvested],
        one row per county-year-crop.
    """
    df = pd.read_parquet(
        NASS_PATH,
        columns=["fips", "year", "crop", "acres_harvested"],
    )
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["fips"].str[:2]

    # Filter state, year, and crop
    mask = (
        df["state_fips"].isin(state_fips)
        & (df["year"] >= year_min)
        & (df["year"] <= year_max)
    )
    if crops is not None:
        mask &= df["crop"].isin(crops)
    df = df[mask].copy()

    # Deduplicate: max acres_harvested per fips/year/crop
    df = df.groupby(["fips", "year", "crop"], as_index=False)["acres_harvested"].max()

    # Drop nulls and zeros
    df = df.dropna(subset=["acres_harvested"])
    df = df[df["acres_harvested"] > 0]

    logger.info(
        f"Loaded NASS acreage: {len(df)} rows, "
