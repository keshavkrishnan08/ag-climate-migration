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

    Args:
        lookup: County name/state lookup from build_county_lookup().

    Returns:
        Clean DataFrame ready for CSV export.

    Raises:
        FileNotFoundError: If either parquet source is missing.
    """
    cols_in = [
        "fips", "year", "crop", "scenario",
        "yield_projected", "yield_baseline", "climate_impact_bu",
        "yield_p10", "yield_p90", "acres_harvested",
    ]
    df245 = pd.read_parquet(
        os.path.join(DATA_PROJ, "yield_projections_SSP245.parquet"),
        columns=cols_in,
    )
    df370 = pd.read_parquet(
        os.path.join(DATA_PROJ, "yield_projections_SSP370.parquet"),
        columns=cols_in,
    )
    df = pd.concat([df245, df370], ignore_index=True)

    df = df.rename(columns={
        "yield_projected": "yield_projected_bu_acre",
        "yield_baseline": "yield_baseline_bu_acre",
        "climate_impact_bu": "climate_impact_bu_acre",
    })
