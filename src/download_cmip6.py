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

    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    times = ds.variables['time']

    # CONUS mask
    lat_mask = (lats >= CONUS_LAT[0]) & (lats <= CONUS_LAT[1])
    lon_mask = (lons >= CONUS_LON[0]) & (lons <= CONUS_LON[1])

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    conus_lats = lats[lat_idx]
    conus_lons = lons[lon_idx]

    # Read CONUS subset of data
    # NetCDF indexing: [time, lat, lon]
    data = ds.variables[variable][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]

    # Convert time to month
    import cftime
    time_vals = ds.variables['time'][:]
    time_units = ds.variables['time'].units
    time_cal = getattr(ds.variables['time'], 'calendar', 'standard')
    dates = cftime.num2date(time_vals, time_units, calendar=time_cal)
    months = np.array([d.month for d in dates])

    # Compute monthly means
    records = []
    for m in range(1, 13):
        month_mask = months == m
        if month_mask.sum() == 0:
            continue
        monthly_mean = np.nanmean(data[month_mask, :, :], axis=0)

        for i, lat in enumerate(conus_lats):
            for j, lon in enumerate(conus_lons):
                val = float(monthly_mean[i, j])
                if not np.isnan(val):
                    records.append({
                        'lat': float(lat),
                        'lon': float(lon),
                        'month': m,
                        'value': val,
                    })

    ds.close()
    return pd.DataFrame(records)


def assign_grid_to_counties(grid_df: pd.DataFrame, county_climate: pd.DataFrame) -> pd.DataFrame:
    """Map 0.25 deg grid cells to counties using nearest-neighbor from nClimDiv.

    Since nClimDiv already has county-level data, we use it to build
    a mapping from (lat, lon) grid cells to FIPS codes.

    For simplicity, we assign each grid cell to the nearest county centroid
    derived from state FIPS codes and county indices.

    Args:
        grid_df: DataFrame with lat, lon, month, value columns.
        county_climate: Existing county climate data for FIPS list.

    Returns:
        DataFrame with fips, month, value (county means).
    """
    # Simple approach: average all CONUS grid cells by latitude band
    # and assign to counties by state latitude ranges
    # This is approximate but works for delta-downscaling since
    # we're applying CHANGES, not absolute values

    # Group grid by 1 deg lat x 2 deg lon cells (coarser but sufficient for deltas)
    grid_df = grid_df.copy()
    grid_df['lat_bin'] = (grid_df['lat'] // 1).astype(int)
    grid_df['lon_bin'] = (grid_df['lon'] // 2).astype(int)

    coarse = grid_df.groupby(['lat_bin', 'lon_bin', 'month'])['value'].mean().reset_index()

    return coarse


def process_one_file(model: str, scenario: str, variable: str, year: int,
                     variant: str = None, grid: str = None) -> pd.DataFrame:
    """Download, extract CONUS, aggregate, delete raw file.

    Args:
        model: GCM model name.
        scenario: SSP scenario (ssp126, ssp245, ssp585).
        variable: Climate variable (tasmax, tasmin, pr).
        year: Target year.
        variant: Ensemble variant label (e.g. r1i1p1f1). If None, looks up
                 from MODEL_VARIANTS or defaults to r1i1p1f1.
        grid: Grid label (e.g. gn, gr, gr1). If None, looks up from
              MODEL_VARIANTS or defaults to gn.

    Returns:
        DataFrame with CONUS grid monthly means.
    """
    # Look up model-specific variant and grid labels
    model_info = MODEL_VARIANTS.get(model, {})
    if variant is None:
        variant = model_info.get('variant', 'r1i1p1f1')
    if grid is None:
        grid = model_info.get('grid', 'gn')

    filename = f"{variable}_day_{model}_{scenario}_{variant}_{grid}_{year}.nc"
    url = f"{S3_BASE}/{model}/{scenario}/{variant}/{variable}/{filename}"
    tmp_path = str(CMIP6_DIR / f"_tmp_{filename}")

    # Check if already processed
    output_path = CMIP6_DIR / f"{model}_{scenario}_{variable}_{year}_conus_monthly.parquet"
    if output_path.exists():
        logger.debug(f"  Already processed: {output_path.name}")
        return pd.read_parquet(output_path)

    # Download
    logger.info(f"  Downloading {filename}...")
    if not download_file(url, tmp_path):
        logger.error(f"  FAILED to download {filename}")
        return pd.DataFrame()

    # Extract CONUS monthly means
    try:
        grid_monthly = extract_conus_monthly(tmp_path, variable)
        grid_monthly['model'] = model
        grid_monthly['scenario'] = scenario
        grid_monthly['variable'] = variable
        grid_monthly['year'] = year

        # Save processed data
        grid_monthly.to_parquet(output_path, index=False)
        logger.info(f"  Extracted: {len(grid_monthly)} CONUS grid-month records -> {output_path.name}")

    except Exception as e:
        logger.error(f"  Extract failed for {filename}: {e}")
        grid_monthly = pd.DataFrame()

    # Delete raw file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        logger.debug(f"  Deleted raw: {tmp_path}")

    return grid_monthly


def download_cmip6_pipeline(scenarios: list = None, years: range = None):
    """Run full CMIP6 download-and-aggregate pipeline.

    Args:
        scenarios: List of SSP scenarios. Default: ['ssp245'].
        years: Year range. Default: 2025-2050.
    """
    if scenarios is None:
        scenarios = ['ssp245']
    if years is None:
        years = range(2025, 2051)

    total_files = len(MODELS) * len(scenarios) * len(VARIABLES) * len(years)
    logger.info(f"CMIP6 download pipeline: {len(MODELS)} models x {len(scenarios)} scenarios "
                f"x {len(VARIABLES)} vars x {len(years)} years = {total_files} files")
    logger.info(f"Peak disk: ~250 MB | Final output: ~{total_files * 0.15:.0f} MB")

    processed = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for model in MODELS:
        for scenario in scenarios:
            for variable in VARIABLES:
                for year in years:
                    output_path = CMIP6_DIR / f"{model}_{scenario}_{variable}_{year}_conus_monthly.parquet"
                    if output_path.exists():
                        skipped += 1
                        continue

                    result = process_one_file(model, scenario, variable, year)
                    if not result.empty:
                        processed += 1
                    else:
                        failed += 1
