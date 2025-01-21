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
