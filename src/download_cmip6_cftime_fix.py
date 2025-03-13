"""
Fix download for models using non-standard calendars (CESM2 and HadGEM3-GC31-LL).

CESM2 uses DatetimeNoLeap; HadGEM3-GC31-LL uses Datetime360Day.
Both require using cftime objects directly rather than converting to pandas DatetimeIndex.

Outputs same parquet format as download_cmip6_pangeo.py.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs
import cftime
from pathlib import Path
import time

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent.parent
CMIP6_DIR  = BASE / "data/raw/cmip6"
CMIP6_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 24.0, 50.0
LON_MIN, LON_MAX = 235.0, 295.0

YEARS = list(range(2025, 2051))
SCENARIO = "ssp245"

MODEL_CONFIG = {
    "CESM2": {
        "member": "r10i1p1f1",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/NCAR/CESM2/ssp245/r10i1p1f1/Amon",
        "vars": {
            "tasmax": "tasmax/gn/v20200528",
            "tasmin": "tasmin/gn/v20200528",
            "pr":     "pr/gn/v20200528",
        },
    },
    "HadGEM3-GC31-LL": {
        "member": "r1i1p1f3",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/MOHC/HadGEM3-GC31-LL/ssp245/r1i1p1f3/Amon",
        "vars": {
            "tasmax": "tasmax/gn/v20190908",
            "tasmin": "tasmin/gn/v20190908",
            "pr":     "pr/gn/v20190908",
        },
    },
}

print("Initializing anonymous GCS filesystem...")
fs = gcsfs.GCSFileSystem(token="anon")


def get_year_from_cftime(t) -> int:
    """Extract year from any cftime or numpy datetime object."""
    if hasattr(t, "year"):
        return t.year
    # numpy datetime64
    return pd.Timestamp(t).year


def get_month_from_cftime(t) -> int:
    """Extract month from any cftime or numpy datetime object."""
    if hasattr(t, "month"):
        return t.month
    return pd.Timestamp(t).month


def load_zarr_cftime(zarr_path: str, var: str) -> xr.DataArray:
    """
    Load a CMIP6 zarr store that uses a non-standard calendar.
    Uses xr.open_zarr with use_cftime=True and avoids pd.DatetimeIndex conversion.

    Args:
        zarr_path: GCS path to zarr store
        var: Variable name

    Returns:
        DataArray sliced to CONUS lat/lon and 2025-2050.
    """
    store = fs.get_mapper(zarr_path)
    ds = xr.open_zarr(store, consolidated=True)
    da = ds[var]

    # Decode time to cftime objects (safe for non-standard calendars)
    # xarray does this automatically; we just can't call pd.DatetimeIndex on it

    # Subset time using xarray's cftime-aware selection
    da = da.sel(time=slice("2025", "2050"))

    # Standardize lon to 0-360
    lons = da.lon.values
    if lons.min() < 0:
        da = da.assign_coords(lon=(da.lon % 360))
        da = da.sortby("lon")

    # Subset to CONUS
    da = da.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX)
    )

    return da


def da_to_annual_parquets_cftime(
    da: xr.DataArray,
    model: str,
    var: str,
    out_dir: Path
) -> list:
    """
    Convert a cftime DataArray to per-year parquet files.

    Handles non-standard calendars by using cftime .year and .month attributes
    instead of pd.DatetimeIndex.

    Args:
        da: DataArray with dims (time, lat, lon), cftime time coordinates
        model: Model name
        var: Variable name
        out_dir: Output directory

    Returns:
        List of written Path objects.
    """
    # Extract year/month from cftime objects directly
    time_vals = da.time.values   # array of cftime objects
    years_arr  = np.array([get_year_from_cftime(t) for t in time_vals])
    months_arr = np.array([get_month_from_cftime(t) for t in time_vals])

    lats = da.lat.values
    lons = da.lon.values
    written = []

    for year in YEARS:
        out_path = out_dir / f"{model}_ssp245_{var}_{year}_conus_monthly.parquet"
        if out_path.exists():
            print(f"    [skip] {out_path.name} already exists")
            written.append(out_path)
            continue

        yr_mask = years_arr == year
        if not yr_mask.any():
            print(f"    [warn] No data for {model}/{var}/{year}")
            continue

        da_yr = da.isel(time=yr_mask)
        months_yr = months_arr[yr_mask]

        # Load into memory
        arr = da_yr.values   # shape: (n_months_this_year, nlat, nlon)

        rows = []
        for mi in range(arr.shape[0]):
            flat_vals = arr[mi].ravel()
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            month_df = pd.DataFrame({
                "lat":   lat_grid.ravel(),
                "lon":   lon_grid.ravel(),
                "month": int(months_yr[mi]),
                "value": flat_vals,
            })
            rows.append(month_df)

        df = pd.concat(rows, ignore_index=True)
        df["model"]    = model
        df["scenario"] = SCENARIO
        df["variable"] = var
        df["year"]     = year
        df = df.dropna(subset=["value"])

        df.to_parquet(out_path, index=False)
        written.append(out_path)
        print(f"    Saved {out_path.name}  ({len(df):,} rows)")

    return written


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

results_log = {}

for model, cfg in MODEL_CONFIG.items():
    print(f"\n{'='*60}")
    print(f"Model: {model}  (member: {cfg['member']}, cftime mode)")
    print(f"{'='*60}")
    results_log[model] = {}

    for var in ["tasmax", "tasmin", "pr"]:
        zarr_path = f"{cfg['zarr_root']}/{cfg['vars'][var]}"
        print(f"\n  Variable: {var}")
        print(f"  Zarr: {zarr_path}")

        t0 = time.time()
        try:
            da = load_zarr_cftime(zarr_path, var)
            print(f"  Loaded: shape={da.shape}, "
                  f"lat=[{float(da.lat.min()):.1f},{float(da.lat.max()):.1f}], "
                  f"lon=[{float(da.lon.min()):.1f},{float(da.lon.max()):.1f}]")

            written = da_to_annual_parquets_cftime(da, model, var, CMIP6_DIR)
            elapsed = time.time() - t0
            results_log[model][var] = f"ok ({len(written)} files, {elapsed:.0f}s)"

        except Exception as e:
            import traceback
            elapsed = time.time() - t0
            msg = f"error: {type(e).__name__}: {e}"
            print(f"  ERROR: {msg}")
            traceback.print_exc()
            results_log[model][var] = msg


