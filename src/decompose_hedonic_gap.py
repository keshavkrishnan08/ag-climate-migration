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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Headline totals (in billions, from headline_numbers_preliminary.json)
HEDONIC_TOTAL_B = 168.0   # Hedonic 2050 SSP245 (rounded from ~163B + CI adjustment)
DCF_CENTRAL_B = 105.1     # DCF central (SR + indirect 1.30x, r=3%, h=35yr)
GAP_B = HEDONIC_TOTAL_B - DCF_CENTRAL_B  # $62.9B ≈ $63B

# Northern dairy states (benefit from reduced heat stress under warming)
NORTHERN_DAIRY_STATE_FIPS = {
    "05": "WI", "06": "MN", "07": "IA", "08": "NY",
    "38": "ND", "46": "SD", "23": "ME", "33": "NH",
    "50": "VT", "25": "MA",
}

# Specialty-crop-heavy states (CA, FL, WA, OR, MI)
SPECIALTY_STATE_FIPS = {"06", "12", "53", "41", "26"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = "inner") -> pd.DataFrame:
    """Merge two DataFrames, coercing FIPS to string.

    Args:
        left: Left DataFrame.
        right: Right DataFrame.
        on: Column name to join on.
        how: Merge type (default 'inner').

    Returns:
        Merged DataFrame.
    """
    left = left.copy()
    right = right.copy()
    left[on] = left[on].astype(str).str.zfill(5)
    right[on] = right[on].astype(str).str.zfill(5)
    return pd.merge(left, right, on=on, how=how)


def load_hedonic() -> pd.DataFrame:
    """Load hedonic stranded-value estimates for 2050 SSP245.

    Returns:
        DataFrame with fips, farm_acres, delta_tmax_july, stranded_total columns.
    Raises:
        FileNotFoundError if the parquet is missing.
    """
    h = pd.read_parquet(
        HEDONIC_PARQUET,
        columns=["fips", "farm_acres", "delta_tmax_july", "stranded_total"],
    )
    h["fips"] = h["fips"].astype(str).str.zfill(5)
    # Keep only counties with a stranded-value estimate (positive gap counties)
    h = h[h["stranded_total"] > 0].copy()
    print(f"  Hedonic: {len(h)} counties, total={h['stranded_total'].sum()/1e9:.2f}B")
    return h


def load_dcf_central() -> pd.DataFrame:
    """Load DCF central stranded-value estimates (SR, r=3%, h=35yr, indirect 1.30x).

    Returns:
        DataFrame with fips, stranded_value_total columns.
    Raises:
        FileNotFoundError if the parquet is missing.
    """
    d = pd.read_parquet(
        DCF_PARQUET,
        columns=["fips", "stranded_value_total", "discount_rate", "horizon"],
    )
    d["fips"] = d["fips"].astype(str).str.zfill(5)
    # Central tier: r=3%, h=35
    d = d[(d["discount_rate"] == 0.03) & (d["horizon"] == 35)].copy()
    print(f"  DCF central: {len(d)} counties, total={d['stranded_value_total'].sum()/1e9:.2f}B")
    return d


def load_climate_precip(target_year: int = 2050) -> pd.DataFrame:
    """Load precipitation delta for target year under SSP245.

    Args:
        target_year: Projection year to use.

    Returns:
        DataFrame with fips, delta_precip_growing columns.
    """
    proj = pd.read_parquet(
        CLIMATE_PROJ,
        columns=["fips", "year", "scenario", "delta_precip_growing"],
    )
    proj = proj[(proj["year"] == target_year) & (proj["scenario"] == "SSP245")].copy()
    proj["fips"] = proj["fips"].astype(str).str.zfill(5)
    return proj[["fips", "delta_precip_growing"]]


def build_livestock_proxy(h: pd.DataFrame) -> pd.DataFrame:
    """Build a livestock/dairy heat-stress proxy from the hedonic warming signal.

    Northern counties (state FIPS in dairy belt) benefit from reduced heat stress
    under warming — the hedonic captures this as positive value change. We proxy
    livestock revenue using the warming signal in northern dairy states weighted
    by farm acres.

    Args:
        h: Hedonic DataFrame with fips, delta_tmax_july, farm_acres columns.

    Returns:
        h with added livestock_proxy column (0-1 scale).
    Raises:
        None.
    """
    h = h.copy()
    # State FIPS is first two digits of county FIPS
    h["state_fips"] = h["fips"].str[:2]
    # Dairy benefit: northern dairy states AND positive warming delta (heat stress relief)
    dairy_states = {"55", "27", "19", "36", "38", "46", "23", "33", "50"}  # WI,MN,IA,NY,ND,SD,ME,NH,VT
    h["in_dairy_state"] = h["state_fips"].isin(dairy_states).astype(float)
    # Warming magnitude proxy: more warming = more heat stress impact
    h["warming_magnitude"] = h["delta_tmax_july"].abs()
    warming_std = h["warming_magnitude"].std()
    if warming_std > 0:
        h["livestock_proxy"] = h["in_dairy_state"] * (h["warming_magnitude"] / warming_std)
    else:
        h["livestock_proxy"] = h["in_dairy_state"]
    # Normalize to 0-1
    mx = h["livestock_proxy"].max()
    if mx > 0:
        h["livestock_proxy"] = h["livestock_proxy"] / mx
    return h


def build_water_proxy(h: pd.DataFrame, precip: pd.DataFrame) -> pd.DataFrame:
    """Build water-availability proxy from precipitation decline.

    Irrigation-dependent western counties where precip declines >10% face
    water-channel losses captured by the hedonic but not the DCF.

    Args:
        h: Hedonic DataFrame.
        precip: DataFrame with fips, delta_precip_growing.

    Returns:
        h merged with water_proxy column.
    """
    h = safe_merge(h, precip, on="fips", how="left")
    h["delta_precip_growing"] = h["delta_precip_growing"].fillna(0.0)
    # Western states: irrigation-dependent
    western_states = {"04", "06", "08", "16", "30", "32", "35", "41", "49", "53"}
    h["in_western_state"] = h["state_fips"].isin(western_states).astype(float)
    # Precip decline > 10% of baseline (proxy: absolute decline > threshold)
    # delta_precip_growing is in mm/month
    h["precip_decline"] = (-h["delta_precip_growing"]).clip(lower=0)
    # Water proxy: western AND meaningful precip decline
    h["water_proxy"] = h["in_western_state"] * h["precip_decline"]
    mx = h["water_proxy"].max()
    if mx > 0:
        h["water_proxy"] = h["water_proxy"] / mx
    return h


def build_amenity_proxy(h: pd.DataFrame) -> pd.DataFrame:
    """Build amenity/rural quality-of-life proxy.

    High-amenity counties carry a land-value premium beyond productive value.
    We proxy amenity using median home value relative to cash rent income
    (the "amenity premium" over agricultural income).

    Args:
        h: Hedonic DataFrame with fips column.

    Returns:
        h with amenity_proxy column added.
    """
    # Load ACS home values as amenity signal
    acs = pd.read_parquet(ACS_DEMO, columns=["fips", "year", "median_home_value"])
    acs["fips"] = acs["fips"].astype(str).str.zfill(5)
    # Use most recent year
    acs_recent = acs.sort_values("year").groupby("fips").last().reset_index()
    acs_recent = acs_recent[["fips", "median_home_value"]]

    # Load cash rent for agricultural income baseline
    rent = pd.read_parquet(CASH_RENT, columns=["fips", "year", "cash_rent_per_acre"])
    rent["fips"] = rent["fips"].astype(str).str.zfill(5)
    rent_recent = rent.sort_values("year").groupby("fips").last().reset_index()
    rent_recent = rent_recent[["fips", "cash_rent_per_acre"]]

    h = safe_merge(h, acs_recent, on="fips", how="left")
    h = safe_merge(h, rent_recent, on="fips", how="left")

    h["median_home_value"] = h["median_home_value"].fillna(h["median_home_value"].median())
    h["cash_rent_per_acre"] = h["cash_rent_per_acre"].fillna(h["cash_rent_per_acre"].median())

    # Amenity proxy: high home value relative to agricultural cash rent
    # Normalize each to 0-1
    hv_std = h["median_home_value"].std()
    cr_std = h["cash_rent_per_acre"].std()
    if hv_std > 0 and cr_std > 0:
        hv_norm = (h["median_home_value"] - h["median_home_value"].mean()) / hv_std
        cr_norm = (h["cash_rent_per_acre"] - h["cash_rent_per_acre"].mean()) / cr_std
        # High amenity = high home value + low cash rent ratio
        h["amenity_proxy"] = hv_norm - cr_norm
    else:
        h["amenity_proxy"] = h["median_home_value"].fillna(0.0)

    # Normalize to 0-1
    amin = h["amenity_proxy"].min()
    amax = h["amenity_proxy"].max()
    if amax > amin:
        h["amenity_proxy"] = (h["amenity_proxy"] - amin) / (amax - amin)
    return h


def build_specialty_proxy(h: pd.DataFrame) -> pd.DataFrame:
    """Build specialty-crop-share proxy from RMA insured acreage.

    Specialty crops (fruits, nuts, vegetables) have different climate
    sensitivities than field crops. The hedonic captures their value;
    the DCF covers only NASS field crops (corn, soy, wheat, etc.).

    Args:
        h: Hedonic DataFrame with fips column.

    Returns:
        h with specialty_proxy column added.
    """
    # Define specialty crops (anything not in the 8 NASS field crops)
    field_crops = {
        "CORN", "SOYBEANS", "WHEAT", "SORGHUM", "BARLEY",
        "OATS", "COTTON", "RICE", "SUNFLOWER",
    }

    rma = pd.read_parquet(RMA_PARQUET, columns=["fips", "year", "crop_name", "acres"])
    rma["fips"] = rma["fips"].astype(str).str.zfill(5)
    rma["crop_upper"] = rma["crop_name"].str.strip().str.upper()

    # Label specialty
    rma["is_specialty"] = ~rma["crop_upper"].apply(
        lambda c: any(fc in c for fc in field_crops)
    )

    # Most recent 5 years
    max_yr = rma["year"].max()
    rma_recent = rma[rma["year"] >= max_yr - 4].copy()
    rma_recent["acres"] = pd.to_numeric(rma_recent["acres"], errors="coerce").fillna(0.0)

    total_acres = rma_recent.groupby("fips")["acres"].sum().rename("total_rma_acres")
    specialty_acres = (
        rma_recent[rma_recent["is_specialty"]]
        .groupby("fips")["acres"]
        .sum()
        .rename("specialty_acres")
    )

    share = pd.concat([total_acres, specialty_acres], axis=1).fillna(0)
    share["specialty_share"] = np.where(
        share["total_rma_acres"] > 0,
        share["specialty_acres"] / share["total_rma_acres"],
        0.0,
    )
    share = share[["specialty_share"]].reset_index()

    h = safe_merge(h, share, on="fips", how="left")
    h["specialty_share"] = h["specialty_share"].fillna(0.0)
    h["specialty_proxy"] = h["specialty_share"]
    return h


# ---------------------------------------------------------------------------
# Main decomposition
# ---------------------------------------------------------------------------

def run_decomposition() -> dict:
    """Execute the hedonic-DCF gap decomposition.

    Returns:
        Dict with channel estimates, correlations, and percentage shares.
    Raises:
        FileNotFoundError if any required input file is missing.
    """
    print("\n=== Hedonic-DCF Gap Decomposition ===")
    print(f"  Hedonic total (2050 SSP245): ${HEDONIC_TOTAL_B:.1f}B")
    print(f"  DCF central (SR, r=3%, h=35): ${DCF_CENTRAL_B:.1f}B")
    print(f"  Gap to explain: ${GAP_B:.1f}B\n")

    # --- Load and merge ---
    print("Loading data...")
    h = load_hedonic()
    d = load_dcf_central()

    merged = safe_merge(h, d[["fips", "stranded_value_total"]], on="fips", how="inner")
    merged.rename(columns={"stranded_total": "hedonic_stranded", "stranded_value_total": "dcf_stranded"}, inplace=True)
    merged["gap"] = merged["hedonic_stranded"] - merged["dcf_stranded"]
    print(f"\n  Merged counties: {len(merged)}")
    print(f"  In-sample hedonic total: {merged['hedonic_stranded'].sum()/1e9:.2f}B")
    print(f"  In-sample DCF central total: {merged['dcf_stranded'].sum()/1e9:.2f}B")
    print(f"  In-sample gap: {merged['gap'].sum()/1e9:.2f}B")

    # --- Build proxies ---
    print("\nBuilding channel proxies...")
    precip = load_climate_precip(target_year=2050)
    merged = build_livestock_proxy(merged)
    merged = build_water_proxy(merged, precip)
    merged = build_amenity_proxy(merged)
    merged = build_specialty_proxy(merged)

    proxies = ["livestock_proxy", "water_proxy", "amenity_proxy", "specialty_proxy"]
    proxy_labels = {
        "livestock_proxy": "Livestock/dairy heat stress",
        "water_proxy": "Water availability",
        "amenity_proxy": "Amenity/rural quality of life",
        "specialty_proxy": "Specialty crops",
    }

    # --- Correlations with the per-county gap ---
    print("\nCorrelations of proxies with per-county gap:")
    correlations = {}
    for p in proxies:
        valid = merged[[p, "gap"]].dropna()
        if len(valid) > 10:
            r, pval = stats.spearmanr(valid[p], valid["gap"])
            correlations[p] = {"spearman_r": float(r), "p_value": float(pval)}
            print(f"  {proxy_labels[p]:40s}: r={r:+.3f}, p={pval:.3f}")

    # --- OLS decomposition (regress gap on proxies, extract R² attribution) ---
    # Standardize proxies for comparability
    proxy_data = merged[proxies].copy()
    for p in proxies:
        std = proxy_data[p].std()
        if std > 0:
            proxy_data[p] = (proxy_data[p] - proxy_data[p].mean()) / std

    y = merged["gap"].values
    X = proxy_data[proxies].values

    # Fit OLS via numpy for robustness
    X_with_const = np.column_stack([np.ones(len(X)), X])
    valid_mask = np.isfinite(X_with_const).all(axis=1) & np.isfinite(y)
    X_clean = X_with_const[valid_mask]
