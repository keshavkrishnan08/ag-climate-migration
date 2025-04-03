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
