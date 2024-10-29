"""
Build county-level climate projections from CMIP6 gridded data using the delta method.

Units:
  PRISM baseline  : Fahrenheit (tmax, tmin), mm/month (precip)
  CMIP6 tasmax/min: Kelvin  → convert delta to °F before adding to PRISM
  CMIP6 pr        : kg/m²/s → convert to mm/month, then take delta

Delta method:
  delta = GCM(target_year) - GCM(2025-2030_mean)
  projected = PRISM_1981-2010_baseline + delta

Workflow:
  1. Load county centroids (CONUS only)
  2. PRISM 1981-2010 baseline per county
  3. Build CMIP6 grid → nearest county lookup
  4. Load GCM data: reference period (2025-2030) + 5 rep years (2030,35,40,45,50)
  5. Delta method + ensemble stats at each rep year
  6. Linear interpolation to annual 2025-2050
  7. Save parquet + print summary
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
CMIP6_DIR   = BASE / "data/raw/cmip6"
PRISM_PATH  = BASE / "data/raw/prism/county_climate_monthly.parquet"
GAZETTE_PATH= BASE / "data/raw/census/2023_Gaz_counties_national.txt"
OUT_PATH    = BASE / "data/projections/county_climate_projections.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
GCMS            = [
    # Original 5 (r1i1p1f1, standard calendar)
    "ACCESS-CM2", "GFDL-ESM4", "MIROC6", "MPI-ESM1-2-HR", "NorESM2-MM",
    # New 5 (Pangeo Cloud via anonymous GCS zarr)
    # CESM2: r10i1p1f1 (no r1i1p1f1 for ssp245 in Pangeo)
    # CNRM-CM6-1: r1i1p1f2 (uses f2 physics variant)
    # HadGEM3-GC31-LL: r1i1p1f3 (AAER forcing; 360-day calendar handled at download)
    # IPSL-CM6A-LR: r1i1p1f1
    # MRI-ESM2-0: r1i1p1f1
    "CESM2", "CNRM-CM6-1", "HadGEM3-GC31-LL", "IPSL-CM6A-LR", "MRI-ESM2-0",
]
REP_YEARS       = [2030, 2035, 2040, 2045, 2050]
REF_YEARS       = list(range(2025, 2031))   # 2025–2030 GCM reference
GROW_MONTHS     = [5, 6, 7, 8, 9]           # May–Sep
JULY            = 7
BASELINE_Y1, BASELINE_Y2 = 1981, 2010
SCENARIO        = "SSP245"
SECS_PER_MONTH  = 30.44 * 86400             # ≈ 2,630,016 s  (pr conversion)
EXCLUDE_STATES  = {"02", "15", "72", "78", "66", "60", "69"}  # AK HI PR VI GU AS CNMI


def k_delta_to_f(delta_k: np.ndarray) -> np.ndarray:
    """Temperature *difference* in K (≡°C) → °F.  No offset, just scale."""
    return delta_k * 9.0 / 5.0


def pr_flux_to_mm_month(flux: np.ndarray) -> np.ndarray:
    """kg m⁻² s⁻¹ → mm month⁻¹ (using 30.44-day month)."""
    return flux * SECS_PER_MONTH


# ═══════════════════════════════════════════════════════════════════════════════
# 1. County centroids
# ═══════════════════════════════════════════════════════════════════════════════
print("Step 1 — Loading county centroids …")
gaz = pd.read_csv(GAZETTE_PATH, sep="\t", dtype=str)
gaz.columns = gaz.columns.str.strip()          # trailing whitespace in headers

gaz["fips"] = gaz["GEOID"].str.zfill(5)
gaz["lat"]  = pd.to_numeric(gaz["INTPTLAT"],  errors="coerce")
gaz["lon"]  = pd.to_numeric(gaz["INTPTLONG"], errors="coerce")

gaz = gaz[~gaz["fips"].str[:2].isin(EXCLUDE_STATES)].copy()
gaz = gaz[["fips", "lat", "lon"]].dropna().reset_index(drop=True)
print(f"  {len(gaz):,} CONUS counties")

county_fips  = gaz["fips"].values
county_lats  = gaz["lat"].values
county_lons  = gaz["lon"].values       # negative for Western Hemisphere


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PRISM 1981-2010 baseline
# ═══════════════════════════════════════════════════════════════════════════════
print("Step 2 — Computing PRISM 1981–2010 baseline …")
prism = pd.read_parquet(PRISM_PATH)

bl = prism[(prism["year"] >= BASELINE_Y1) & (prism["year"] <= BASELINE_Y2)].copy()

tmax_g_cols   = [f"tmax_m{m:02d}"   for m in GROW_MONTHS]  # 5 cols
tmin_g_cols   = [f"tmin_m{m:02d}"   for m in GROW_MONTHS]
pr_g_cols     = [f"precip_m{m:02d}" for m in GROW_MONTHS]

bl["_tmax_grow"]   = bl[tmax_g_cols].mean(axis=1)
bl["_tmin_grow"]   = bl[tmin_g_cols].mean(axis=1)
bl["_precip_grow"] = bl[pr_g_cols].sum(axis=1)   # total precip per growing season

baseline = (bl.groupby("fips")
              .agg(
                  tmax_july_bl     = ("tmax_m07",     "mean"),
                  tmax_growing_bl  = ("_tmax_grow",   "mean"),
                  precip_growing_bl= ("_precip_grow", "mean"),
                  tmin_growing_bl  = ("_tmin_grow",   "mean"),
              )
              .reset_index())

print(f"  Baseline for {len(baseline):,} counties  (PRISM = °F / mm/month)")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CMIP6 grid → county nearest-neighbour lookup
# ═══════════════════════════════════════════════════════════════════════════════
print("Step 3 — Building per-model CMIP6 grid → county lookups …")

# Convert county lons to 0-360 for matching
county_lons_360 = county_lons % 360         # wraps -180..0 → 180..360

# Build a separate nearest-neighbour lookup for each GCM (they have different grids)
_nn_keys_per_gcm = {}
for gcm in GCMS:
    ref_path = CMIP6_DIR / f"{gcm}_ssp245_tasmax_2025_conus_monthly.parquet"
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

    nn_lat_gcm = grid_lats_gcm[idx]
    nn_lon_gcm = grid_lons_gcm[idx]
    _nn_keys_per_gcm[gcm] = list(zip(nn_lat_gcm.tolist(), nn_lon_gcm.tolist()))
    print(f"  {gcm}: {len(grid_pts):,} grid points matched to {len(county_fips):,} counties")

# Default nn_keys for backward compatibility (used by _load_county_gcm)
nn_keys = _nn_keys_per_gcm.get(GCMS[0], [])
print("  Grid matching done.")


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: load one CMIP6 file → county-level growing-season & July values
# ═══════════════════════════════════════════════════════════════════════════════

def _load_county_gcm(gcm: str, var: str, year: int) -> dict:
    """
    Load a CMIP6 parquet file and extract county-level aggregates.

    For temperature (tasmax, tasmin): growing-season mean (months 5-9) and July mean.
    For precipitation (pr): growing-season mean flux [kg m⁻² s⁻¹], averaged over the
    5 growing months (so each entry is the mean monthly flux; we convert later).

    Returns:
        dict with keys 'growing' (array len n_counties) and 'july' (array or None for pr).
    """
    path = CMIP6_DIR / f"{gcm}_ssp245_{var}_{year}_conus_monthly.parquet"
    df   = pd.read_parquet(path)

    # Use per-model grid lookup (different GCMs have different resolutions)
    gcm_nn_keys = _nn_keys_per_gcm.get(gcm, nn_keys)

    # Growing season subset
    grow = df[df["month"].isin(GROW_MONTHS)].copy()
    grow["_key"] = list(zip(grow["lat"].tolist(), grow["lon"].tolist()))

    # Aggregate across months per grid point
    # For temp: mean; for pr: mean (we keep it as mean flux for now)
    grow_agg = grow.groupby("_key")["value"].mean()

    grow_vals = grow_agg.reindex(gcm_nn_keys).values.astype(float)

    if var == "pr":
        return {"growing": grow_vals, "july": None}

    # July for temperature variables
    july = df[df["month"] == JULY].copy()
    july["_key"] = list(zip(july["lat"].tolist(), july["lon"].tolist()))
    july_agg  = july.groupby("_key")["value"].mean()
    july_vals = july_agg.reindex(gcm_nn_keys).values.astype(float)

    return {"growing": grow_vals, "july": july_vals}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Load GCM data — reference period (2025-2030) and 5 rep years
# ═══════════════════════════════════════════════════════════════════════════════
print("\nStep 4 — Loading CMIP6 files …")
print(f"  Reference period: {REF_YEARS[0]}–{REF_YEARS[-1]}")

# ref_data[gcm][var] = {"growing": array(n_counties), "july": array or None}
ref_data = {}
for gcm in GCMS:
    print(f"    {gcm} reference …")
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

# rep_data[gcm][var][yr] = {"growing": array, "july": array or None}
rep_data = {}
for gcm in GCMS:
    print(f"    {gcm} rep years …")
    rep_data[gcm] = {}
    for var in ["tasmax", "tasmin", "pr"]:
        rep_data[gcm][var] = {}
        for yr in REP_YEARS:
            rep_data[gcm][var][yr] = _load_county_gcm(gcm, var, yr)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Delta method at each representative year
# ═══════════════════════════════════════════════════════════════════════════════
print("\nStep 5 — Delta method …")

bl_idx = baseline.set_index("fips")

# Build a DataFrame: one row per (county, rep_year)
rep_records = []

for yr in REP_YEARS:
    # Collect per-GCM deltas — shape (n_gcms, n_counties)
    D_tmax_july    = np.full((len(GCMS), len(county_fips)), np.nan)
    D_tmax_grow    = np.full((len(GCMS), len(county_fips)), np.nan)
    D_tmin_grow    = np.full((len(GCMS), len(county_fips)), np.nan)
    D_pr_grow      = np.full((len(GCMS), len(county_fips)), np.nan)   # mm/month delta

    for gi, gcm in enumerate(GCMS):
        # ---- tasmax ----
        ref_tx_j = ref_data[gcm]["tasmax"]["july"]      # K, shape (n_counties,)
        tgt_tx_j = rep_data[gcm]["tasmax"][yr]["july"]
        D_tmax_july[gi] = k_delta_to_f(tgt_tx_j - ref_tx_j)

        ref_tx_g = ref_data[gcm]["tasmax"]["growing"]
        tgt_tx_g = rep_data[gcm]["tasmax"][yr]["growing"]
        D_tmax_grow[gi] = k_delta_to_f(tgt_tx_g - ref_tx_g)

        # ---- tasmin ----
        ref_tn_g = ref_data[gcm]["tasmin"]["growing"]
        tgt_tn_g = rep_data[gcm]["tasmin"][yr]["growing"]
        D_tmin_grow[gi] = k_delta_to_f(tgt_tn_g - ref_tn_g)

        # ---- pr ----
        # Both ref and tgt are mean monthly flux [kg m⁻² s⁻¹] averaged over grow months
        # Convert each to mm/month, then take delta
        ref_pr_mm = pr_flux_to_mm_month(ref_data[gcm]["pr"]["growing"])
        tgt_pr_mm = pr_flux_to_mm_month(rep_data[gcm]["pr"][yr]["growing"])
        D_pr_grow[gi] = tgt_pr_mm - ref_pr_mm

    # Ensemble statistics across GCMs (axis=0)
    med_tmax_j = np.nanmedian(D_tmax_july, axis=0)
    med_tmax_g = np.nanmedian(D_tmax_grow, axis=0)
    med_tmin_g = np.nanmedian(D_tmin_grow, axis=0)
    med_pr_g   = np.nanmedian(D_pr_grow,   axis=0)

    p10_tmax_j = np.nanpercentile(D_tmax_july, 10, axis=0)
    p90_tmax_j = np.nanpercentile(D_tmax_july, 90, axis=0)
    n_gcms_arr = np.sum(~np.isnan(D_tmax_july), axis=0).astype(int)

    for i, fips in enumerate(county_fips):
        if fips not in bl_idx.index:
            continue   # not in PRISM (tiny territories)
        bl_row = bl_idx.loc[fips]

        rep_records.append({
            "fips":     fips,
            "year":     yr,
