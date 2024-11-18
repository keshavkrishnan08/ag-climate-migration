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
        dlon = county_lons_360[s:e, None] - grid_lons_gcm[None, :]
        idx[s:e] = np.argmin(dlat**2 + dlon**2, axis=1)

    _nn_keys_per_gcm[gcm] = list(zip(
        grid_lats_gcm[idx].tolist(),
        grid_lons_gcm[idx].tolist()
    ))
    print(f"  {gcm}: {len(grid_pts):,} grid points -> {len(county_fips):,} counties")

nn_keys = _nn_keys_per_gcm.get(GCMS[0], [])
print("  Grid matching done.")


def _load_county_gcm(gcm, var, year):
    """Load a CMIP6 SSP370 parquet file and extract county-level aggregates.

    For temperature variables: growing-season mean (months 5-9) and July mean.
    For precipitation: growing-season mean flux in kg/m2/s.

    Args:
        gcm: GCM name matching the file prefix.
        var: Variable name (tasmax, tasmin, or pr).
        year: Target year (2025-2050).

    Returns:
        Dict with 'growing' array (len n_counties) and 'july' array or None for pr.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
    """
    path = CMIP6_DIR / f"{gcm}_ssp370_{var}_{year}_conus_monthly.parquet"
    df   = pd.read_parquet(path)

    gcm_nn_keys = _nn_keys_per_gcm.get(gcm, nn_keys)

    grow = df[df["month"].isin(GROW_MONTHS)].copy()
    grow["_key"] = list(zip(grow["lat"].tolist(), grow["lon"].tolist()))
    grow_agg  = grow.groupby("_key")["value"].mean()
    grow_vals = grow_agg.reindex(gcm_nn_keys).values.astype(float)

    if var == "pr":
        return {"growing": grow_vals, "july": None}

    july = df[df["month"] == JULY].copy()
    july["_key"] = list(zip(july["lat"].tolist(), july["lon"].tolist()))
    july_agg  = july.groupby("_key")["value"].mean()
    july_vals = july_agg.reindex(gcm_nn_keys).values.astype(float)

    return {"growing": grow_vals, "july": july_vals}


# Step 4: Load GCM data
print("\nStep 4 - Loading CMIP6 SSP370 files ...")
print(f"  Reference period: {REF_YEARS[0]}-{REF_YEARS[-1]}")

ref_data = {}
for gcm in GCMS:
    if gcm not in _nn_keys_per_gcm:
        print(f"    {gcm}: skipped (no grid lookup)")
        continue
    print(f"    {gcm} reference ...")
    ref_data[gcm] = {}
    for var in ["tasmax", "tasmin", "pr"]:
        stacks_grow = []
        stacks_july = []
        for yr in REF_YEARS:
            res = _load_county_gcm(gcm, var, yr)
            stacks_grow.append(res["growing"])
            if res["july"] is not None:
                stacks_july.append(res["july"])
        ref_data[gcm][var] = {
            "growing": np.nanmean(stacks_grow, axis=0),
            "july":    np.nanmean(stacks_july, axis=0) if stacks_july else None,
        }

print(f"  Representative years: {REP_YEARS}")

rep_data = {}
for gcm in GCMS:
    if gcm not in _nn_keys_per_gcm:
        continue
    print(f"    {gcm} rep years ...")
    rep_data[gcm] = {}
    for var in ["tasmax", "tasmin", "pr"]:
        rep_data[gcm][var] = {}
        for yr in REP_YEARS:
            rep_data[gcm][var][yr] = _load_county_gcm(gcm, var, yr)


# Step 5: Delta method
print("\nStep 5 - Delta method ...")

ACTIVE_GCMS = [g for g in GCMS if g in ref_data]
bl_idx = baseline.set_index("fips")
rep_records = []

for yr in REP_YEARS:
    D_tmax_july = np.full((len(ACTIVE_GCMS), len(county_fips)), np.nan)
    D_tmax_grow = np.full((len(ACTIVE_GCMS), len(county_fips)), np.nan)
    D_tmin_grow = np.full((len(ACTIVE_GCMS), len(county_fips)), np.nan)
    D_pr_grow   = np.full((len(ACTIVE_GCMS), len(county_fips)), np.nan)

    for gi, gcm in enumerate(ACTIVE_GCMS):
        ref_tx_j = ref_data[gcm]["tasmax"]["july"]
        tgt_tx_j = rep_data[gcm]["tasmax"][yr]["july"]
        D_tmax_july[gi] = k_delta_to_f(tgt_tx_j - ref_tx_j)

        ref_tx_g = ref_data[gcm]["tasmax"]["growing"]
        tgt_tx_g = rep_data[gcm]["tasmax"][yr]["growing"]
        D_tmax_grow[gi] = k_delta_to_f(tgt_tx_g - ref_tx_g)

        ref_tn_g = ref_data[gcm]["tasmin"]["growing"]
        tgt_tn_g = rep_data[gcm]["tasmin"][yr]["growing"]
        D_tmin_grow[gi] = k_delta_to_f(tgt_tn_g - ref_tn_g)

        ref_pr_mm = pr_flux_to_mm_month(ref_data[gcm]["pr"]["growing"])
        tgt_pr_mm = pr_flux_to_mm_month(rep_data[gcm]["pr"][yr]["growing"])
        D_pr_grow[gi]   = tgt_pr_mm - ref_pr_mm

    med_tmax_j = np.nanmedian(D_tmax_july, axis=0)
    med_tmax_g = np.nanmedian(D_tmax_grow, axis=0)
    med_tmin_g = np.nanmedian(D_tmin_grow, axis=0)
    med_pr_g   = np.nanmedian(D_pr_grow,   axis=0)

    p10_tmax_j = np.nanpercentile(D_tmax_july, 10, axis=0)
    p90_tmax_j = np.nanpercentile(D_tmax_july, 90, axis=0)
    n_gcms_arr = np.sum(~np.isnan(D_tmax_july), axis=0).astype(int)

    for i, fips in enumerate(county_fips):
        if fips not in bl_idx.index:
            continue
        bl_row = bl_idx.loc[fips]

        rep_records.append({
            "fips":     fips,
            "year":     yr,
            "scenario": SCENARIO,
            "tmax_july_projected":      bl_row["tmax_july_bl"]      + med_tmax_j[i],
            "tmax_growing_projected":   bl_row["tmax_growing_bl"]   + med_tmax_g[i],
            "precip_growing_projected": bl_row["precip_growing_bl"] + med_pr_g[i],
            "tmin_growing_projected":   bl_row["tmin_growing_bl"]   + med_tmin_g[i],
            "delta_tmax_july":          med_tmax_j[i],
            "delta_tmax_growing":       med_tmax_g[i],
            "delta_precip_growing":     med_pr_g[i],
            "delta_tmin_growing":       med_tmin_g[i],
            "tmax_july_p10": bl_row["tmax_july_bl"] + p10_tmax_j[i],
            "tmax_july_p90": bl_row["tmax_july_bl"] + p90_tmax_j[i],
            "n_gcms": int(n_gcms_arr[i]),
        })

proj_rep = pd.DataFrame(rep_records)
print(f"  Rep-year table: {proj_rep.shape}")


# Step 6: Interpolate to annual 2025-2050
print("\nStep 6 - Interpolating to annual resolution ...")

ALL_YEARS = list(range(2025, 2051))

INTERP_COLS = [
    "tmax_july_projected", "tmax_growing_projected",
    "precip_growing_projected", "tmin_growing_projected",
    "delta_tmax_july", "delta_tmax_growing",
    "delta_precip_growing", "delta_tmin_growing",
    "tmax_july_p10", "tmax_july_p90",
]

anchor_records = []
for fips in proj_rep["fips"].unique():
    if fips not in bl_idx.index:
        continue
    bl_row = bl_idx.loc[fips]
    anchor_records.append({
        "fips":     fips,
        "year":     2025,
        "scenario": SCENARIO,
        "tmax_july_projected":      bl_row["tmax_july_bl"],
        "tmax_growing_projected":   bl_row["tmax_growing_bl"],
        "precip_growing_projected": bl_row["precip_growing_bl"],
        "tmin_growing_projected":   bl_row["tmin_growing_bl"],
        "delta_tmax_july":    0.0,
        "delta_tmax_growing": 0.0,
        "delta_precip_growing": 0.0,
        "delta_tmin_growing": 0.0,
        "tmax_july_p10": bl_row["tmax_july_bl"],
        "tmax_july_p90": bl_row["tmax_july_bl"],
        "n_gcms": len(ACTIVE_GCMS),
    })

anchor_df = pd.DataFrame(anchor_records)
pivot_df  = pd.concat([anchor_df, proj_rep[proj_rep["year"] > 2025]], ignore_index=True)
pivot_df  = pivot_df.sort_values(["fips", "year"]).reset_index(drop=True)

KNOT_YEARS = [2025] + REP_YEARS

all_records = []
unique_fips = pivot_df["fips"].unique()

for fips in unique_fips:
    sub        = pivot_df[pivot_df["fips"] == fips].set_index("year")
    n_gcms_val = int(sub["n_gcms"].max())

    for yr in ALL_YEARS:
        if yr in sub.index:
            row = sub.loc[yr]
            out = {"fips": fips, "year": yr, "scenario": SCENARIO, "n_gcms": n_gcms_val}
            for col in INTERP_COLS:
                out[col] = float(row[col])
        else:
            lo    = max(k for k in KNOT_YEARS if k <= yr)
            hi    = min(k for k in KNOT_YEARS if k >= yr)
            alpha = (yr - lo) / (hi - lo)
            row_lo= sub.loc[lo]
            row_hi= sub.loc[hi]
            out   = {"fips": fips, "year": yr, "scenario": SCENARIO, "n_gcms": n_gcms_val}
            for col in INTERP_COLS:
                out[col] = float(row_lo[col]) + alpha * (float(row_hi[col]) - float(row_lo[col]))
        all_records.append(out)

result = pd.DataFrame(all_records)
result["n_gcms"] = result["n_gcms"].astype(int)
print(f"  Final table: {result.shape}")


# Step 7: Save
result.to_parquet(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")


# Step 8: Summary
