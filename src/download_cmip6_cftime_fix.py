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

