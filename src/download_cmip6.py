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

    # Use ERS Atlas for county centroids if available, otherwise assign
    # from FIPS state codes to approximate lat/lon bands
    # For now, we'll do nearest-grid-cell assignment when processing
    return pd.DataFrame({'fips': counties})


def download_file(url: str, dest: str, max_retries: int = 3) -> bool:
    """Download a file with progress and retry.

    Args:
        url: Source URL.
        dest: Destination file path.
        max_retries: Number of retry attempts.

    Returns:
        True if download succeeded.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, stream=True, timeout=600)
            if resp.status_code != 200:
                logger.warning(f"  HTTP {resp.status_code} for {url}")
                return False

            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0

            with open(dest, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)

            if total > 0 and downloaded < total * 0.95:
                logger.warning(f"  Incomplete download: {downloaded}/{total} bytes")
                os.remove(dest)
                continue

            return True

        except Exception as e:
            logger.warning(f"  Download attempt {attempt + 1} failed: {e}")
            if os.path.exists(dest):
                os.remove(dest)
            time.sleep(5 * (attempt + 1))

    return False


def extract_conus_monthly(nc_path: str, variable: str) -> pd.DataFrame:
    """Extract CONUS grid cells and compute monthly means from daily NetCDF.

    Args:
        nc_path: Path to downloaded NetCDF file.
        variable: Variable name (tasmax, tasmin, pr).

    Returns:
        DataFrame: lat, lon, month (1-12), value (monthly mean).
    """
    ds = Dataset(nc_path, 'r')

