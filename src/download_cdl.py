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
