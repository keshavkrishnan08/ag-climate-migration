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

    df = add_county_info(df, lookup)
    df = df.sort_values(["fips", "crop", "scenario", "year"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# File 2: county_climate_projections.csv
# ---------------------------------------------------------------------------

def build_climate_projections(lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Combine SSP245 and SSP370 county-level climate projections.
    Temperatures stored as °F (tmax columns) consistent with source data.

    Args:
        lookup: County name/state lookup.

    Returns:
        Clean DataFrame ready for CSV export.
    """
    keep = [
        "fips", "year", "scenario", "n_gcms",
        "tmax_july_projected", "delta_tmax_july",
        "precip_growing_projected", "delta_precip_growing",
        "tmax_july_p10", "tmax_july_p90",
    ]
    df245 = pd.read_parquet(
        os.path.join(DATA_PROJ, "county_climate_projections.parquet"),
        columns=keep,
    )
    df370 = pd.read_parquet(
        os.path.join(DATA_PROJ, "county_climate_projections_ssp370.parquet"),
        columns=keep,
    )
    df = pd.concat([df245, df370], ignore_index=True)

    df = df.rename(columns={
        "tmax_july_projected": "tmax_july_projected_F",
        "delta_tmax_july": "delta_tmax_july_F",
        "precip_growing_projected": "precip_growing_projected_mm",
        "delta_precip_growing": "delta_precip_mm",
    })

    df = add_county_info(df, lookup)
    df = df.sort_values(["fips", "scenario", "year"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# File 3: county_stranded_assets.csv
# ---------------------------------------------------------------------------

def build_stranded_assets(lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Merge DCF stranded values (conservative=SSP245 4%, central=SSP370 4%),
    hedonic stranded values (2050 horizon), and cap-rate overvaluation
    into a single county-level table.

    Args:
        lookup: County name/state lookup.

    Returns:
        One row per county with stranded asset valuations.
    """
    sa_dir = os.path.join(RESULTS, "stranded_assets")

    # DCF conservative = SSP245 central discount (4%)
    dcf_cons = pd.read_parquet(
        os.path.join(sa_dir, "stranded_national_SSP245.parquet"),
        columns=["fips", "stranded_value_total", "stranded_value_per_acre",
                 "land_value_per_acre", "stranded_fraction", "total_acres"],
    ).rename(columns={
        "stranded_value_total": "stranded_dcf_conservative_usd",
        "stranded_value_per_acre": "stranded_dcf_conservative_per_acre_usd",
    })

    # DCF central = SSP370 (higher warming)
    dcf_cent = pd.read_parquet(
        os.path.join(sa_dir, "stranded_national_SSP370.parquet"),
        columns=["fips", "stranded_value_total"],
    ).rename(columns={"stranded_value_total": "stranded_dcf_central_usd"})

    # Hedonic (2050 horizon, SSP245)
    hedonic = pd.read_parquet(
        os.path.join(sa_dir, "hedonic_stranded.parquet"),
        columns=["fips", "stranded_total", "target_year", "scenario"],
    )
    hedonic_2050 = (
        hedonic[hedonic["target_year"] == 2050]
        .query("scenario == 'SSP245'")
        [["fips", "stranded_total"]]
        .rename(columns={"stranded_total": "stranded_hedonic_usd"})
    )

    # Merge
    df = (
        dcf_cons
        .merge(dcf_cent, on="fips", how="left")
        .merge(hedonic_2050, on="fips", how="left")
    )

    df = df.rename(columns={
        "stranded_dcf_conservative_per_acre_usd": "stranded_per_acre_usd",
        "total_acres": "total_farm_acres",
    })

    # Select and order columns
    # Rename land value col before column selection (source has no _usd suffix)
    df = df.rename(columns={"land_value_per_acre": "land_value_per_acre_usd"})

    df = df[
        [
            "fips",
            "stranded_dcf_conservative_usd",
            "stranded_dcf_central_usd",
            "stranded_hedonic_usd",
            "stranded_per_acre_usd",
            "stranded_fraction",
            "land_value_per_acre_usd",
            "total_farm_acres",
        ]
    ].copy()

    df = add_county_info(df, lookup)
    df = df.sort_values("fips").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# File 4: county_decline_indicators.csv
# ---------------------------------------------------------------------------

def build_decline_indicators(lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Combine historical cascade decline signals (2005-2023) with tipping-year
    estimates from two independent methods (own IV, Feng et al. 2010).

    Args:
        lookup: County name/state lookup.
