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

