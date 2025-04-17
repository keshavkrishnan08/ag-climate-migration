"""Targeted CMIP6 download — only what the paper needs.

Downloads 5 diverse GCMs x ssp245 x 3 vars x milestone years.
Then linearly interpolates annual values for the projection pipeline.

Usage:
    python src/download_cmip6_targeted.py
"""

import sys
from pathlib import Path

# Reuse the main download machinery
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_cmip6 import process_one_file, CMIP6_DIR, VARIABLES, MODEL_VARIANTS
from loguru import logger
import time
import pandas as pd
import numpy as np
import os

# 5 GCMs spanning warm/cool/wet/dry for ensemble spread.
# All verified to have tasmax/tasmin/pr on the NASA NEX-GDDP-CMIP6 S3 bucket.
# CESM2 excluded: only has tas (mean temp), not tasmax/tasmin.
PRIORITY_MODELS = [
    'ACCESS-CM2',       # warm/wet (Australia)       — r1i1p1f1, gn
    'GFDL-ESM4',        # moderate (NOAA)            — r1i1p1f1, gr1
    'MIROC6',           # warm (Japan)               — r1i1p1f1, gn
    'MPI-ESM1-2-HR',    # moderate/cool (Germany)    — r1i1p1f1, gn
    'NorESM2-MM',       # moderate (Norway)          — r1i1p1f1, gn
]

# Milestone years — enough for interpolation + key decadal snapshots
MILESTONE_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]


def download_targeted():
    """Download only the priority models x milestone years.

    Uses model-specific variant labels and grid labels from MODEL_VARIANTS
    to construct the correct S3 URLs for each GCM.
    """
    total = len(PRIORITY_MODELS) * len(VARIABLES) * len(MILESTONE_YEARS)
    logger.info(f"Targeted CMIP6: {len(PRIORITY_MODELS)} models x {len(VARIABLES)} vars "
                f"x {len(MILESTONE_YEARS)} years = {total} files")

    # Log the variant/grid info for each model
    for model in PRIORITY_MODELS:
        info = MODEL_VARIANTS.get(model, {})
        logger.info(f"  {model}: variant={info.get('variant', '???')}, "
