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
