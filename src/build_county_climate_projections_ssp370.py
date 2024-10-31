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
