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
    )
    df = df.merge(total, on=["fips", "year"], how="left")
    df["share"] = df["acres_harvested"] / df["total_acres"]
    return df


def compute_pair_switching(
    shares_wide: pd.DataFrame,
    from_crop: str,
    to_crop: str,
    threshold_pp: float = THRESHOLD_PP,
) -> pd.Series:
    """Compute switching rate for one (from→to) pair across all county-years.

    Logic:
        - Year-over-year change in `from_crop` share: delta_from = share_from[t] - share_from[t-1]
        - Year-over-year change in `to_crop` share:   delta_to   = share_to[t]   - share_to[t-1]
        - If delta_from < -threshold_pp AND delta_to > 0:
              switching_rate = delta_to   (bounded to [0, 1])
          else 0.

    Args:
        shares_wide: Wide DataFrame indexed by (fips, year) with one column
            per crop containing the crop's share. Missing crops have NaN.
        from_crop: Name of the source crop (e.g. "corn").
        to_crop: Name of the destination crop (e.g. "soybeans").
        threshold_pp: Minimum drop in from_crop share (in fraction, not pct)
            to signal a potential switch.

    Returns:
        Series of switching rates indexed by (fips, year), name =
        f"switch_{from_crop}_to_{to_crop}".
    """
    col_name = f"switch_{from_crop}_to_{to_crop}"

    # Extract the two share columns; fill missing with 0 (crop not present)
    from_share = shares_wide[from_crop] if from_crop in shares_wide.columns else None
    to_share   = shares_wide[to_crop]   if to_crop   in shares_wide.columns else None

    if from_share is None or to_share is None:
        # One of the crops has no data at all — return zeros
        result = pd.Series(0.0, index=shares_wide.index, name=col_name)
        return result

    # Fill NaN with 0 (county-year had 0 acres of that crop)
    from_share = from_share.fillna(0.0)
    to_share   = to_share.fillna(0.0)

    # Year-over-year delta within each county
    # groupby on level 0 = fips, then diff on year-sorted data
    delta_from = (
        from_share
        .groupby(level="fips", sort=True)
        .diff()
    )
    delta_to = (
        to_share
        .groupby(level="fips", sort=True)
        .diff()
    )

    # Switching condition
    switched = (delta_from < -threshold_pp) & (delta_to > 0)

    rate = pd.Series(0.0, index=shares_wide.index, name=col_name)
    rate[switched] = delta_to[switched].clip(lower=0.0, upper=1.0)

    return rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Build switching_rates.parquet and print diagnostics."""
    print("Loading NASS data …")
    df = load_nass(RAW_NASS)
    print(f"  Loaded {len(df):,} rows | {df['fips'].nunique():,} counties | "
          f"years {df['year'].min()}–{df['year'].max()}")

    print("Computing county-year acreage shares …")
    df = compute_county_shares(df)

    # Pivot to wide: index=(fips,year), columns=crop, values=share
    print("Pivoting to wide format …")
    shares_wide = (
        df.pivot_table(
            index=["fips", "year"],
            columns="crop",
            values="share",
            aggfunc="first",
        )
        .sort_index()
    )
    shares_wide.columns.name = None  # tidy up

    print(f"  Wide shape: {shares_wide.shape}  (counties×years × crops)")

    # Compute each pair's switching rate
    switch_cols = {}
    for from_crop, to_crop in PAIRS:
        col = f"switch_{from_crop}_to_{to_crop}"
        print(f"  Computing {col} …")
        switch_cols[col] = compute_pair_switching(shares_wide, from_crop, to_crop)

    # Assemble output
    out = pd.DataFrame(switch_cols, index=shares_wide.index).reset_index()
    out["fips"] = out["fips"].str.zfill(5)
    out["year"] = out["year"].astype(int)

    # Sort for reproducibility
    out = out.sort_values(["fips", "year"]).reset_index(drop=True)

    # Save
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")

    # --- Diagnostics ---
    print(f"\nShape: {out.shape}")
    print("\nSample rows (first 8):")
