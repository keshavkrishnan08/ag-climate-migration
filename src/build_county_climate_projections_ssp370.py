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

