"""
Build county-level climate projections from CMIP6 SSP3-7.0 gridded data.

Uses the same delta method as build_county_climate_projections.py but reads
from data/raw/cmip6_ssp370/ and writes to
data/projections/county_climate_projections_ssp370.parquet.

Units, delta method, and interpolation logic are identical to the SSP2-4.5 version.
GCM substitutions vs. SSP2-4.5:
  - MPI-ESM1-2-HR -> MPI-ESM1-2-LR (HR not in Pangeo ssp370)
  - HadGEM3-GC31-LL -> UKESM1-0-LL (same Met Office family; LL not in Pangeo ssp370)
  - NorESM2-MM -> dropped (tasmax/tasmin not available in Pangeo ssp370 Amon)
  Net: 9 GCMs for SSP370 vs. 10 for SSP245.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
BASE        = Path(__file__).resolve().parent.parent
CMIP6_DIR   = BASE / "data/raw/cmip6_ssp370"
PRISM_PATH  = BASE / "data/raw/prism/county_climate_monthly.parquet"
GAZETTE_PATH= BASE / "data/raw/census/2023_Gaz_counties_national.txt"
OUT_PATH    = BASE / "data/projections/county_climate_projections_ssp370.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Constants
GCMS = [
    "ACCESS-CM2",
    "GFDL-ESM4",
    "MIROC6",
    "MPI-ESM1-2-LR",
    "CNRM-CM6-1",
    "IPSL-CM6A-LR",
    "MRI-ESM2-0",
    "CESM2",
    "UKESM1-0-LL",
]

REP_YEARS       = [2030, 2035, 2040, 2045, 2050]
REF_YEARS       = list(range(2025, 2031))
GROW_MONTHS     = [5, 6, 7, 8, 9]
JULY            = 7
BASELINE_Y1, BASELINE_Y2 = 1981, 2010
SCENARIO        = "SSP370"
SECS_PER_MONTH  = 30.44 * 86400
EXCLUDE_STATES  = {"02", "15", "72", "78", "66", "60", "69"}


def k_delta_to_f(delta_k):
    """Temperature difference in K to F (scale only, no offset).

    Args:
        delta_k: ndarray of temperature deltas in K/degC.

    Returns:
        ndarray converted to Fahrenheit scale.
    """
    return delta_k * 9.0 / 5.0


def pr_flux_to_mm_month(flux):
    """Convert kg m^-2 s^-1 to mm/month using 30.44-day month.

    Args:
        flux: ndarray of precipitation flux.

    Returns:
        ndarray of monthly precipitation in mm.
    """
    return flux * SECS_PER_MONTH


# Step 1: County centroids
print("Step 1 - Loading county centroids ...")
gaz = pd.read_csv(GAZETTE_PATH, sep="\t", dtype=str)
gaz.columns = gaz.columns.str.strip()

gaz["fips"] = gaz["GEOID"].str.zfill(5)
gaz["lat"]  = pd.to_numeric(gaz["INTPTLAT"],  errors="coerce")
gaz["lon"]  = pd.to_numeric(gaz["INTPTLONG"], errors="coerce")

gaz = gaz[~gaz["fips"].str[:2].isin(EXCLUDE_STATES)].copy()
gaz = gaz[["fips", "lat", "lon"]].dropna().reset_index(drop=True)
print(f"  {len(gaz):,} CONUS counties")

county_fips     = gaz["fips"].values
county_lats     = gaz["lat"].values
county_lons     = gaz["lon"].values


# Step 2: PRISM 1981-2010 baseline
print("Step 2 - Computing PRISM 1981-2010 baseline ...")
prism = pd.read_parquet(PRISM_PATH)

bl = prism[(prism["year"] >= BASELINE_Y1) & (prism["year"] <= BASELINE_Y2)].copy()

tmax_g_cols   = [f"tmax_m{m:02d}"   for m in GROW_MONTHS]
tmin_g_cols   = [f"tmin_m{m:02d}"   for m in GROW_MONTHS]
pr_g_cols     = [f"precip_m{m:02d}" for m in GROW_MONTHS]

bl["_tmax_grow"]   = bl[tmax_g_cols].mean(axis=1)
bl["_tmin_grow"]   = bl[tmin_g_cols].mean(axis=1)
bl["_precip_grow"] = bl[pr_g_cols].sum(axis=1)

baseline = (bl.groupby("fips")
              .agg(
                  tmax_july_bl     = ("tmax_m07",     "mean"),
                  tmax_growing_bl  = ("_tmax_grow",   "mean"),
                  precip_growing_bl= ("_precip_grow", "mean"),
                  tmin_growing_bl  = ("_tmin_grow",   "mean"),
              )
              .reset_index())

print(f"  Baseline for {len(baseline):,} counties")


# Step 3: CMIP6 grid to county nearest-neighbour lookup
print("Step 3 - Building per-model CMIP6 grid -> county lookups ...")

county_lons_360 = county_lons % 360

_nn_keys_per_gcm = {}
for gcm in GCMS:
    ref_path = CMIP6_DIR / f"{gcm}_ssp370_tasmax_2025_conus_monthly.parquet"
    if not ref_path.exists():
        print(f"  {gcm}: reference file not found, skipping")
        continue
    _ref = pd.read_parquet(ref_path, columns=["lat", "lon"])
    grid_pts = _ref[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    grid_lats_gcm = grid_pts["lat"].values
    grid_lons_gcm = grid_pts["lon"].values

    BATCH = 300
    idx = np.empty(len(county_fips), dtype=np.int64)
    for s in range(0, len(county_fips), BATCH):
        e = min(s + BATCH, len(county_fips))
        dlat = county_lats[s:e, None] - grid_lats_gcm[None, :]
