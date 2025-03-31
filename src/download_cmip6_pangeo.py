"""
Download 5 missing CMIP6 GCMs from Pangeo Cloud (anonymous GCS access).

Models: CESM2, CNRM-CM6-1, HadGEM3-GC31-LL, IPSL-CM6A-LR, MRI-ESM2-0
Scenario: SSP2-4.5
Variables: tasmax, tasmin, pr
Years: 2025-2050 (extracted from 2015-2100 zarr stores)

Outputs:
    data/raw/cmip6/{MODEL}_ssp245_{VAR}_{YEAR}_conus_monthly.parquet
    Columns: lat, lon, month, value, model, scenario, variable, year
    Coordinates: lat 24-50 N, lon 235-295 (0-360 convention)

Notes:
    - CESM2 uses r10i1p1f1 (r1i1p1f1 not in Pangeo for ssp245)
    - CNRM-CM6-1 uses r1i1p1f2 (physics variant; same as CNRM standard)
    - HadGEM3-GC31-LL uses r1i1p1f3 (uses AAER forcing; standard for this model)
    - IPSL and MRI use r1i1p1f1 (standard)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs
from pathlib import Path
import time

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent.parent
CMIP6_DIR  = BASE / "data/raw/cmip6"
CMIP6_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
# CONUS bounding box in 0-360 lon convention
LAT_MIN, LAT_MAX = 24.0, 50.0
LON_MIN, LON_MAX = 235.0, 295.0   # 115W-65W in 0-360

YEARS = list(range(2025, 2051))    # 2025–2050 inclusive
SCENARIO = "ssp245"

# Model → (member_id, zarr_base_path)
MODEL_CONFIG = {
    "CESM2": {
        "member": "r10i1p1f1",
        "institution": "NCAR",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/NCAR/CESM2/ssp245/r10i1p1f1/Amon",
        "vars": {
            "tasmax": "tasmax/gn/v20200528",
            "tasmin": "tasmin/gn/v20200528",
            "pr":     "pr/gn/v20200528",
        },
    },
    "CNRM-CM6-1": {
        "member": "r1i1p1f2",
        "institution": "CNRM-CERFACS",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/CNRM-CERFACS/CNRM-CM6-1/ssp245/r1i1p1f2/Amon",
        "vars": {
            "tasmax": "tasmax/gr/v20190219",
            "tasmin": "tasmin/gr/v20190219",
            "pr":     "pr/gr/v20190219",
        },
    },
    "HadGEM3-GC31-LL": {
        "member": "r1i1p1f3",
        "institution": "MOHC",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/MOHC/HadGEM3-GC31-LL/ssp245/r1i1p1f3/Amon",
        "vars": {
            "tasmax": "tasmax/gn/v20190908",
            "tasmin": "tasmin/gn/v20190908",
            "pr":     "pr/gn/v20190908",
        },
    },
    "IPSL-CM6A-LR": {
        "member": "r1i1p1f1",
        "institution": "IPSL",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp245/r1i1p1f1/Amon",
        "vars": {
            "tasmax": "tasmax/gr/v20190119",
            "tasmin": "tasmin/gr/v20190119",
            "pr":     "pr/gr/v20190119",
        },
    },
    "MRI-ESM2-0": {
        "member": "r1i1p1f1",
        "institution": "MRI",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/MRI/MRI-ESM2-0/ssp245/r1i1p1f1/Amon",
        "vars": {
            "tasmax": "tasmax/gn/v20190222",
            "tasmin": "tasmin/gn/v20190222",
            "pr":     "pr/gn/v20190222",
        },
    },
}

# ── GCS filesystem (anonymous) ─────────────────────────────────────────────────
print("Initializing anonymous GCS filesystem...")
fs = gcsfs.GCSFileSystem(token="anon")


def load_zarr_variable(zarr_path: str, var: str) -> xr.DataArray:
    """
    Load a CMIP6 variable from a GCS zarr store, subset to CONUS.

    Args:
        zarr_path: GCS path like 'gs://cmip6/CMIP6/...'
        var: Variable name ('tasmax', 'tasmin', or 'pr')

    Returns:
        DataArray sliced to CONUS lat/lon and 2025-2050.

    Raises:
        Exception: If zarr store is unreachable or variable missing.
    """
    store = fs.get_mapper(zarr_path)
    ds = xr.open_zarr(store, consolidated=True)

    da = ds[var]

    # Handle 360-day calendars by converting to standard time index
    times = pd.DatetimeIndex(da.time.values)

    # Subset time to 2025-2050
    mask_time = (times.year >= 2025) & (times.year <= 2050)
    da = da.isel(time=mask_time)

    # Standardize lon to 0-360 if needed
    lons = da.lon.values
    if lons.min() < 0:
        # -180..180 → 0..360
        da = da.assign_coords(lon=(da.lon % 360))
        da = da.sortby("lon")

    # Subset to CONUS
    da = da.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX)
    )

    return da


def da_to_annual_parquets(
    da: xr.DataArray,
    model: str,
    var: str,
    out_dir: Path
) -> list[Path]:
    """
    Convert a DataArray (time, lat, lon) to per-year parquet files.

    Args:
        da: DataArray with dims (time, lat, lon), time already subsetted to 2025-2050
        model: Model name for file naming and metadata column
        var: Variable name
        out_dir: Directory to write parquet files

    Returns:
        List of Path objects for written files.

    Raises:
        ValueError: If DataArray has unexpected dimensions.
    """
    times = pd.DatetimeIndex(da.time.values)
    written = []

    for year in YEARS:
        out_path = out_dir / f"{model}_ssp245_{var}_{year}_conus_monthly.parquet"
        if out_path.exists():
            print(f"    [skip] {out_path.name} already exists")
            written.append(out_path)
            continue

        yr_mask = times.year == year
        if not yr_mask.any():
            print(f"    [warn] No data for {model}/{var}/{year}")
            continue

        da_yr = da.isel(time=yr_mask)

        # Load into memory and reshape to long format
        arr = da_yr.values          # shape: (12, nlat, nlon)
        lats = da_yr.lat.values
        lons = da_yr.lon.values

        # months: 1..12 (inferred from time index)
        year_times = pd.DatetimeIndex(da_yr.time.values)
        months = year_times.month.values   # array of month ints

        # Build long-format dataframe
        rows = []
        for mi, month in enumerate(months):
            flat_vals = arr[mi].ravel()          # (nlat*nlon,)
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            lat_flat = lat_grid.ravel()
            lon_flat = lon_grid.ravel()

            month_df = pd.DataFrame({
                "lat":      lat_flat,
                "lon":      lon_flat,
                "month":    month,
                "value":    flat_vals,
            })
            rows.append(month_df)

        df = pd.concat(rows, ignore_index=True)
        df["model"]    = model
        df["scenario"] = SCENARIO
        df["variable"] = var
        df["year"]     = year

        # Drop NaN grid cells (ocean, out-of-bounds)
        df = df.dropna(subset=["value"])

        df.to_parquet(out_path, index=False)
        written.append(out_path)
        print(f"    Saved {out_path.name}  ({len(df):,} rows)")

    return written


# ══════════════════════════════════════════════════════════════════════════════
# Main download loop
# ══════════════════════════════════════════════════════════════════════════════

results_log = {}   # model → {var: "ok" | "error: ..."}

for model, cfg in MODEL_CONFIG.items():
    print(f"\n{'='*60}")
    print(f"Model: {model}  (member: {cfg['member']})")
    print(f"{'='*60}")
    results_log[model] = {}

    for var in ["tasmax", "tasmin", "pr"]:
        zarr_path = f"{cfg['zarr_root']}/{cfg['vars'][var]}"
        print(f"\n  Variable: {var}")
        print(f"  Zarr: {zarr_path}")

        t0 = time.time()
        try:
            da = load_zarr_variable(zarr_path, var)
            print(f"  Loaded: shape={da.shape}, "
                  f"lat=[{float(da.lat.min()):.1f},{float(da.lat.max()):.1f}], "
                  f"lon=[{float(da.lon.min()):.1f},{float(da.lon.max()):.1f}]")

            written = da_to_annual_parquets(da, model, var, CMIP6_DIR)
            elapsed = time.time() - t0
            results_log[model][var] = f"ok ({len(written)} files, {elapsed:.0f}s)"
