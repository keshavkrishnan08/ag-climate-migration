"""Stream-download CDL rasters and compute county switching matrices.

Downloads one year at a time, extracts county-level crop type summaries
using rasterio + county FIPS raster, computes switching rates, deletes raw.
Peak disk: ~2.5 GB. Final output: a few MB.

Usage:
    python src/download_cdl.py                    # 2018-2023 (6 years, recent switching)
    python src/download_cdl.py --years 2008-2023  # full CDL range
"""

import os
import sys
import argparse
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
CDL_DIR = DATA_RAW / 'cdl'
CDL_DIR.mkdir(parents=True, exist_ok=True)

CDL_BASE = "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets"

# CDL crop codes -> our crop names
CDL_CROP_CODES = {
    1: 'corn',
    5: 'soybeans',
    21: 'barley',
    22: 'durum_wheat',
    23: 'wheat_spring',
    24: 'wheat_winter',
    2: 'cotton',
    4: 'sorghum',
    28: 'oats',
    6: 'sunflower',
}

# Primary crops we track for switching
PRIMARY_CROPS = {1, 5, 24, 23, 2, 4, 21, 28}

# Latitude band definitions (approximate row boundaries for EPSG:5070 CDL)
# Based on the transform: row -> lat mapping is roughly linear over CONUS
# Row 0 ~ 51.6N, Row 96522 ~ 25.5N
# We define 5 latitude bands:
#   Northern:  lat >= 45   (rows 0 - ~24000)
#   Upper Mid: 41 <= lat < 45 (rows ~24000 - ~38000)
#   Central:   37 <= lat < 41 (rows ~38000 - ~52000)
#   Lower Mid: 33 <= lat < 37 (rows ~52000 - ~66000)
#   Southern:  lat < 33    (rows ~66000+)
LATITUDE_BANDS = {
    'northern': (0, 24200),       # ~45N and above
    'upper_midwest': (24200, 38400),  # ~41N to ~45N
    'central': (38400, 52600),    # ~37N to ~41N
    'lower_midwest': (52600, 66800),  # ~33N to ~37N
    'southern': (66800, 100000),  # below ~33N
}


def _row_to_lat_band(rows: np.ndarray) -> np.ndarray:
    """Map pixel row indices to latitude band labels.

    Args:
        rows: Array of row indices.

    Returns:
        Array of string labels for each row's latitude band.
    """
    bands = np.empty(len(rows), dtype='U15')
    bands[:] = 'southern'  # default
    for band_name, (row_start, row_end) in LATITUDE_BANDS.items():
        mask = (rows >= row_start) & (rows < row_end)
        bands[mask] = band_name
    return bands


def download_cdl_year(year: int) -> str:
    """Download CDL zip for one year.

    Args:
        year: CDL year to download.

    Returns:
        Path to extracted TIF file, or empty string on failure.
    """
    zip_name = f"{year}_30m_cdls.zip"
    url = f"{CDL_BASE}/{zip_name}"
    zip_path = CDL_DIR / zip_name

    # Check if already extracted
    tif_candidates = list(CDL_DIR.glob(f"*{year}*cdl*.tif")) + list(CDL_DIR.glob(f"*{year}*CDL*.tif"))
    if tif_candidates:
        logger.debug(f"  CDL {year}: already extracted -> {tif_candidates[0].name}")
        return str(tif_candidates[0])

    # Download
    logger.info(f"  Downloading CDL {year} (~2 GB)...")
    try:
        resp = requests.get(url, stream=True, timeout=1800)
        if resp.status_code != 200:
            logger.error(f"  HTTP {resp.status_code} for CDL {year}")
            return ""

        total = int(resp.headers.get('Content-Length', 0))
        downloaded = 0
        with open(zip_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (500 * 1024 * 1024) == 0:
                    pct = downloaded / total * 100 if total else 0
                    logger.info(f"    {downloaded/1e9:.1f} GB / {total/1e9:.1f} GB ({pct:.0f}%)")

    except Exception as e:
        logger.error(f"  CDL {year} download failed: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return ""

    # Extract TIF
    logger.info(f"  Extracting CDL {year}...")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            tif_files = [n for n in zf.namelist() if n.lower().endswith('.tif')]
            if not tif_files:
                logger.error(f"  No TIF found in CDL {year} zip")
                return ""
            zf.extract(tif_files[0], CDL_DIR)
            tif_path = CDL_DIR / tif_files[0]

        # Delete zip to save space
        zip_path.unlink()
        logger.info(f"  Extracted: {tif_path.name}")
        return str(tif_path)

    except Exception as e:
        logger.error(f"  CDL {year} extract failed: {e}")
        return ""


def compute_county_crop_summary_simple(tif_path: str, year: int) -> pd.DataFrame:
    """Compute crop pixel counts by latitude band from CDL raster.

    Reads the raster in horizontal strips to avoid loading the full
    ~14 GB array into memory. Counts pixels of each primary crop within
    each latitude band, providing a spatial breakdown of crop acreage.

    Each 30m pixel = 0.9 hectares = ~2.224 acres.

    Args:
        tif_path: Path to CDL GeoTIFF.
        year: CDL year.

    Returns:
        DataFrame with columns: lat_band, year, crop_code, crop_name,
        pixel_count, est_acres.
    """
    try:
        import rasterio
    except ImportError:
        logger.warning("rasterio not installed -- cannot read CDL raster")
        return pd.DataFrame(columns=[
            'lat_band', 'year', 'crop_code', 'crop_name',
            'pixel_count', 'est_acres'
        ])

    logger.info(f"  Computing crop summary for CDL {year}...")

    # We'll accumulate counts per (lat_band, crop_code)
    primary_list = sorted(PRIMARY_CROPS)
    band_names = list(LATITUDE_BANDS.keys())
    # Initialize count matrix: bands x crops
    counts = {band: np.zeros(256, dtype=np.int64) for band in band_names}

    strip_height = 512  # match the raster block size

    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        logger.info(f"    Raster: {w} x {h}, reading in {strip_height}-row strips")

        for row_start in range(0, h, strip_height):
            row_end = min(row_start + strip_height, h)
            actual_height = row_end - row_start

            # Read strip
            window = rasterio.windows.Window(0, row_start, w, actual_height)
            data = src.read(1, window=window)  # shape: (actual_height, w), dtype uint8

            # Determine which lat band this strip falls in
            strip_mid_row = row_start + actual_height // 2
            band_name = 'southern'
            for bn, (rs, re) in LATITUDE_BANDS.items():
                if rs <= strip_mid_row < re:
                    band_name = bn
                    break

            # Count crop pixels using bincount (fast for uint8)
            flat = data.ravel()
            bc = np.bincount(flat, minlength=256)
            counts[band_name] += bc

            if row_start % (strip_height * 50) == 0 and row_start > 0:
                logger.debug(f"    Processed {row_start}/{h} rows")

    # Build result DataFrame
    rows = []
    acres_per_pixel = 30.0 * 30.0 / 4046.86  # 30m pixel -> acres

    for band_name in band_names:
        for crop_code in primary_list:
            pc = int(counts[band_name][crop_code])
            if pc > 0:
                rows.append({
                    'lat_band': band_name,
                    'year': year,
                    'crop_code': crop_code,
                    'crop_name': CDL_CROP_CODES.get(crop_code, f'code_{crop_code}'),
                    'pixel_count': pc,
                    'est_acres': round(pc * acres_per_pixel, 1),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        total_crop_px = df['pixel_count'].sum()
        logger.info(
            f"    CDL {year}: {total_crop_px:,.0f} primary crop pixels "
            f"({total_crop_px * acres_per_pixel / 1e6:.1f}M acres)"
        )
    return df


def compute_switching_from_cdl_pair(
    tif_year_t: str,
    tif_year_t1: str,
