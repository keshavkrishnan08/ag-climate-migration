"""
Download CMIP6 SSP3-7.0 data from Pangeo Cloud for 9 GCMs.

SSP3-7.0 eliminates the single-scenario limitation in the AgMigration pipeline.
Uses anonymous GCS access to the Pangeo CMIP6 zarr archive — no ESGF account needed.

Availability notes vs. SSP2-4.5:
  - MPI-ESM1-2-HR: not in Pangeo ssp370; substituted with MPI-ESM1-2-LR (same institution)
  - NorESM2-MM: ssp370 data exists in Pangeo but only has tas/pr, no tasmax/tasmin
  - HadGEM3-GC31-LL: not in Pangeo ssp370 at all
  → 7 direct matches + MPI-ESM1-2-LR + UKESM1-0-LL = 9 GCMs total

Calendar notes:
  - CESM2: NoLeap calendar → cftime approach
  - CNRM-CM6-1: standard calendar (proleptic_gregorian) → pd.DatetimeIndex OK
  - UKESM1-0-LL: 360-day calendar → cftime approach
  - All others: standard proleptic_gregorian → pd.DatetimeIndex OK

Outputs:
    data/raw/cmip6_ssp370/{MODEL}_ssp370_{VAR}_{YEAR}_conus_monthly.parquet
    Columns: lat, lon, month, value, model, scenario, variable, year
    Coordinates: lat 24-50 N, lon 235-295 (0-360 convention)
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
BASE      = Path(__file__).resolve().parent.parent
OUT_DIR   = BASE / "data/raw/cmip6_ssp370"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 24.0, 50.0
LON_MIN, LON_MAX = 235.0, 295.0   # 115W–65W in 0-360 convention

YEARS    = list(range(2025, 2051))
SCENARIO = "ssp370"

# Models with standard calendars (pd.DatetimeIndex safe)
MODEL_CONFIG_STANDARD = {
    "ACCESS-CM2": {
        "member":   "r1i1p1f1",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/CSIRO-ARCCSS/ACCESS-CM2/ssp370/r1i1p1f1/Amon",
        "vars": {
            "tasmax": "tasmax/gn/v20191108",
            "tasmin": "tasmin/gn/v20191108",
            "pr":     "pr/gn/v20191108",
        },
    },
    "GFDL-ESM4": {
        # ssp370 uses NoLeap calendar (unlike ssp245 which was standard gregorian);
        # moved to MODEL_CONFIG_CFTIME below.
        # Kept here as a placeholder so the dict ordering is preserved.
        # See MODEL_CONFIG_CFTIME for the actual download config.
        "_skip": True,
        "member":   "r1i1p1f1",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/NOAA-GFDL/GFDL-ESM4/ssp370/r1i1p1f1/Amon",
        "vars": {
            "tasmax": "tasmax/gr1/v20180701",
            "tasmin": "tasmin/gr1/v20180701",
            "pr":     "pr/gr1/v20180701",
        },
    },
    "MIROC6": {
        "member":   "r1i1p1f1",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/MIROC/MIROC6/ssp370/r1i1p1f1/Amon",
        "vars": {
            "tasmax": "tasmax/gn/v20190627",
            "tasmin": "tasmin/gn/v20190627",
            "pr":     "pr/gn/v20190627",
        },
    },
    "MPI-ESM1-2-LR": {
        # MPI-ESM1-2-HR not available in Pangeo ssp370; LR is same institution,
        # same model family, and suitable for ensemble spread characterisation.
        "member":   "r1i1p1f1",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/MPI-M/MPI-ESM1-2-LR/ssp370/r1i1p1f1/Amon",
        "vars": {
            "tasmax": "tasmax/gn/v20190710",
            "tasmin": "tasmin/gn/v20190710",
            "pr":     "pr/gn/v20190710",
        },
    },
    "CNRM-CM6-1": {
        "member":   "r1i1p1f2",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/CNRM-CERFACS/CNRM-CM6-1/ssp370/r1i1p1f2/Amon",
        "vars": {
            "tasmax": "tasmax/gr/v20190219",
            "tasmin": "tasmin/gr/v20190219",
            "pr":     "pr/gr/v20190219",
        },
    },
    "IPSL-CM6A-LR": {
        "member":   "r1i1p1f1",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp370/r1i1p1f1/Amon",
        "vars": {
            "tasmax": "tasmax/gr/v20190119",
            "tasmin": "tasmin/gr/v20190119",
            "pr":     "pr/gr/v20190119",
        },
    },
    "MRI-ESM2-0": {
        "member":   "r1i1p1f1",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/MRI/MRI-ESM2-0/ssp370/r1i1p1f1/Amon",
        "vars": {
            "tasmax": "tasmax/gn/v20190222",
            "tasmin": "tasmin/gn/v20190222",
            "pr":     "pr/gn/v20190222",
        },
    },
}

# Models with non-standard calendars (cftime approach required)
MODEL_CONFIG_CFTIME = {
    "GFDL-ESM4": {
        # NoLeap calendar in ssp370 (verified from Pangeo zarr)
        "member":   "r1i1p1f1",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/NOAA-GFDL/GFDL-ESM4/ssp370/r1i1p1f1/Amon",
        "vars": {
            "tasmax": "tasmax/gr1/v20180701",
            "tasmin": "tasmin/gr1/v20180701",
            "pr":     "pr/gr1/v20180701",
        },
    },
    "CESM2": {
        # NoLeap calendar
        "member":   "r10i1p1f1",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/NCAR/CESM2/ssp370/r10i1p1f1/Amon",
        "vars": {
            "tasmax": "tasmax/gn/v20200528",
            "tasmin": "tasmin/gn/v20200528",
            "pr":     "pr/gn/v20200528",
        },
    },
    "UKESM1-0-LL": {
        # 360-day calendar; substitutes for HadGEM3-GC31-LL (same Met Office family)
        # and fills the NorESM2-MM gap. r1i1p1f2 is MOHC's standard ScenarioMIP member.
        "member":   "r1i1p1f2",
        "zarr_root": "gs://cmip6/CMIP6/ScenarioMIP/MOHC/UKESM1-0-LL/ssp370/r1i1p1f2/Amon",
        "vars": {
            "tasmax": "tasmax/gn/v20190726",
            "tasmin": "tasmin/gn/v20190726",
            "pr":     "pr/gn/v20190510",
        },
    },
}

# ── GCS filesystem ─────────────────────────────────────────────────────────────
print("Initializing anonymous GCS filesystem...")
fs = gcsfs.GCSFileSystem(token="anon")


# ══════════════════════════════════════════════════════════════════════════════
# Standard-calendar helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_zarr_standard(zarr_path: str, var: str) -> xr.DataArray:
    """
    Load a CMIP6 zarr store with a proleptic_gregorian calendar.

    Subsets to CONUS bounding box and 2025-2050 using pd.DatetimeIndex.

    Args:
        zarr_path: GCS path (gs://...) to the zarr store.
        var: Variable name ('tasmax', 'tasmin', or 'pr').

    Returns:
        DataArray with dims (time, lat, lon) sliced to CONUS, 2025-2050.

    Raises:
        Exception: If store is unreachable or calendar is non-standard.
    """
    store = fs.get_mapper(zarr_path)
    ds    = xr.open_zarr(store, consolidated=True)
    da    = ds[var]

    # Time subset via pd.DatetimeIndex
    times    = pd.DatetimeIndex(da.time.values)
    mask     = (times.year >= 2025) & (times.year <= 2050)
    da       = da.isel(time=mask)

    # Standardize lon to 0-360
    if da.lon.values.min() < 0:
        da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")

    # Spatial subset
    da = da.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
    return da


def da_to_annual_parquets(
    da: xr.DataArray,
    model: str,
    var: str,
    out_dir: Path,
) -> list:
    """
    Convert a standard-calendar DataArray to per-year parquet files.

    Args:
        da: DataArray (time, lat, lon) already subsetted to 2025-2050.
        model: Model name for file naming and metadata column.
        var: Variable name.
        out_dir: Directory to write parquet files.

    Returns:
        List of written Path objects.

    Raises:
        ValueError: If the DataArray has unexpected dimension order.
    """
    times   = pd.DatetimeIndex(da.time.values)
    written = []

    for year in YEARS:
        out_path = out_dir / f"{model}_{SCENARIO}_{var}_{year}_conus_monthly.parquet"
        if out_path.exists():
            print(f"    [skip] {out_path.name}")
            written.append(out_path)
            continue

        yr_mask = times.year == year
        if not yr_mask.any():
            print(f"    [warn] No data for {model}/{var}/{year}")
            continue

        da_yr  = da.isel(time=yr_mask)
        arr    = da_yr.values                   # (n_months, nlat, nlon)
        lats   = da_yr.lat.values
        lons   = da_yr.lon.values
        months = pd.DatetimeIndex(da_yr.time.values).month.values

        rows = []
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        lat_flat = lat_grid.ravel()
        lon_flat = lon_grid.ravel()

        for mi, month in enumerate(months):
            rows.append(pd.DataFrame({
                "lat":   lat_flat,
                "lon":   lon_flat,
                "month": int(month),
                "value": arr[mi].ravel(),
            }))

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
# cftime helpers (non-standard calendars: NoLeap, 360-day)
# ══════════════════════════════════════════════════════════════════════════════

def _year_of(t) -> int:
    """Extract year from cftime or numpy datetime object."""
    if hasattr(t, "year"):
        return t.year
    return pd.Timestamp(t).year


def _month_of(t) -> int:
    """Extract month from cftime or numpy datetime object."""
    if hasattr(t, "month"):
        return t.month
    return pd.Timestamp(t).month


def load_zarr_cftime(zarr_path: str, var: str) -> xr.DataArray:
    """
    Load a CMIP6 zarr store with a non-standard calendar (NoLeap or 360-day).

    Uses xarray's string-based time slicing, which works with cftime objects.

    Args:
        zarr_path: GCS path (gs://...) to the zarr store.
        var: Variable name.

    Returns:
        DataArray with dims (time, lat, lon) sliced to CONUS, 2025-2050.

    Raises:
