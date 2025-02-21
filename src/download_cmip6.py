"""Stream-download CMIP6 data from NASA NEX-GDDP-CMIP6 (public S3).

Downloads one file at a time, extracts CONUS, aggregates to county
monthly means, deletes raw file. Peak disk: ~250 MB. Final output: ~350 MB.

Usage:
    python src/download_cmip6.py                    # ssp245 only (primary)
    python src/download_cmip6.py --all-scenarios    # ssp126 + ssp245 + ssp585
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import requests
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
CMIP6_DIR = DATA_RAW / 'cmip6'
CMIP6_DIR.mkdir(parents=True, exist_ok=True)

S3_BASE = "https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com/NEX-GDDP-CMIP6"

MODELS = [
    'ACCESS-CM2', 'CNRM-CM6-1', 'GFDL-ESM4',
    'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC6',
    'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM'
]

# Model-specific variant labels and grid labels for NASA NEX-GDDP-CMIP6.
# Discovered by querying the S3 bucket — not all models use r1i1p1f1 / gn.
MODEL_VARIANTS = {
    'ACCESS-CM2':       {'variant': 'r1i1p1f1', 'grid': 'gn'},
    'CNRM-CM6-1':       {'variant': 'r1i1p1f2', 'grid': 'gr'},
    'GFDL-ESM4':        {'variant': 'r1i1p1f1', 'grid': 'gr1'},
    'HadGEM3-GC31-LL':  {'variant': 'r1i1p1f3', 'grid': 'gn'},
    'IPSL-CM6A-LR':     {'variant': 'r1i1p1f1', 'grid': 'gr'},
    'MIROC6':           {'variant': 'r1i1p1f1', 'grid': 'gn'},
    'MPI-ESM1-2-HR':    {'variant': 'r1i1p1f1', 'grid': 'gn'},
    'MRI-ESM2-0':       {'variant': 'r1i1p1f1', 'grid': 'gn'},
    'NorESM2-MM':       {'variant': 'r1i1p1f1', 'grid': 'gn'},
}

VARIABLES = ['tasmax', 'tasmin', 'pr']

# CONUS bounding box (0-360 longitude convention)
CONUS_LAT = (24.0, 50.0)
CONUS_LON = (235.0, 295.0)  # -125 + 360 = 235, -65 + 360 = 295

# County centroids for spatial aggregation (built from climate data)
def load_county_grid() -> pd.DataFrame:
    """Load county FIPS with approximate grid cell assignments.

    Uses the nClimDiv county climate data we already have to identify
    which 0.25 deg grid cells map to which counties. Approximate but
    sufficient for delta-downscaling.

    Returns:
        DataFrame: fips, lat_idx, lon_idx for nearest grid cell.
    """
    climate = pd.read_parquet(DATA_RAW / 'prism' / 'county_climate_annual.parquet',
                              columns=['fips'])
    counties = climate['fips'].unique()

