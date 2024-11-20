"""
build_published_dataset.py

Assemble 6 publication-ready CSVs from the ag_migration pipeline outputs.
All monetary values in 2023 USD. Temperatures in °F (climate files).
Yields in bushels per acre.

Run from repo root:
    python src/build_published_dataset.py

Outputs (data/published_dataset/):
    county_yield_projections.csv
    county_climate_projections.csv
    county_stranded_assets.csv
    county_decline_indicators.csv
    county_insurance_mispricing.csv
    county_opportunity_frontier.csv
"""

import os
import sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROJ = os.path.join(ROOT, "data", "projections")
DATA_RAW = os.path.join(ROOT, "data", "raw")
RESULTS = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "data", "published_dataset")

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# County name lookup
# ---------------------------------------------------------------------------

def build_county_lookup() -> pd.DataFrame:
    """
    Build a FIPS → (county_name, state) lookup from the Census Gazetteer.

    Returns:
        DataFrame with columns: fips, county_name, state
    """
    gaz_path = os.path.join(DATA_RAW, "census", "2023_Gaz_counties_national.txt")
    gaz = pd.read_csv(gaz_path, sep="\t", dtype={"GEOID": str}, usecols=["GEOID", "NAME", "USPS"])
    gaz = gaz.rename(columns={"GEOID": "fips", "NAME": "county_name", "USPS": "state"})
    gaz["fips"] = gaz["fips"].str.zfill(5)
    return gaz[["fips", "county_name", "state"]].drop_duplicates("fips")


def add_county_info(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join county_name and state onto df using fips.

    Args:
        df: DataFrame with a 'fips' column (5-digit string).
        lookup: Output of build_county_lookup().

    Returns:
        df with county_name and state columns inserted after fips.
    """
    df = df.copy()
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    merged = df.merge(lookup, on="fips", how="left")
    # Re-order so fips, county_name, state come first
    other_cols = [c for c in merged.columns if c not in ("fips", "county_name", "state")]
    return merged[["fips", "county_name", "state"] + other_cols]


# ---------------------------------------------------------------------------
# File 1: county_yield_projections.csv
# ---------------------------------------------------------------------------

def build_yield_projections(lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Combine SSP245 and SSP370 yield projections with county identifiers.
