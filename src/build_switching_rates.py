"""Build county-level crop switching rates from NASS acreage data.

Computes year-over-year share changes as a proxy for crop switching.
For each pair (A→B): when A's share drops >5pp AND B's share rises,
the switching rate equals the increase in B's acreage share.

Output: data/processed/switching_rates.parquet
Columns:
    fips                          (str, 5-digit zero-padded)
    year                          (int)
    switch_corn_to_soybeans       (float, 0-1)
    switch_corn_to_sorghum        (float, 0-1)
    switch_cotton_to_soybeans     (float, 0-1)
    switch_wheat_winter_to_wheat_spring (float, 0-1)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_NASS = PROJECT_ROOT / "data" / "raw" / "nass" / "nass_county_yields.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "switching_rates.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Seed (for reproducibility; no stochastic ops here, kept for consistency)
RNG = np.random.default_rng(42)

# Switching pairs
PAIRS = [
    ("corn",         "soybeans"),
    ("corn",         "sorghum"),
    ("cotton",       "soybeans"),
    ("wheat_winter", "wheat_spring"),
]

# Threshold: A's share must drop by at least this many percentage points
THRESHOLD_PP = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_nass(path: Path) -> pd.DataFrame:
    """Load NASS yields, apply CONUS/year/aggregate filters, dedup.

    Args:
        path: Path to nass_county_yields.parquet.

    Returns:
        Filtered, deduped DataFrame with columns [fips, year, crop,
        acres_harvested].
    """
    df = pd.read_parquet(path, columns=["fips", "year", "crop", "acres_harvested"])

    # --- Filter ---
    # Year >= 1950
    df = df[df["year"] >= 1950]
    # CONUS only: exclude AK (02), HI (15), PR (72)
    df = df[~df["fips"].str[:2].isin(["02", "15", "72"])]
    # Remove state-level aggregates (998) and national/other aggregates (999)
    df = df[~df["fips"].str.endswith(("998", "999"))]

    # Ensure 5-digit zero-padded FIPS
    df["fips"] = df["fips"].str.zfill(5)

    # Dedup per CLAUDE.md spec: groupby(['fips','year','crop']).first()
    df = (
        df.groupby(["fips", "year", "crop"], sort=False)
        .first()
        .reset_index()
    )

    # Drop rows with null acreage (can't compute shares)
    df = df.dropna(subset=["acres_harvested"])
    # Drop negative/zero acreage
    df = df[df["acres_harvested"] > 0]

    return df


def compute_county_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Compute each crop's share of total harvested acres per county-year.

    Args:
        df: Filtered NASS DataFrame with [fips, year, crop, acres_harvested].

    Returns:
        DataFrame with added column `share` (0–1).
    """
    total = (
        df.groupby(["fips", "year"])["acres_harvested"]
        .sum()
        .rename("total_acres")
        .reset_index()
