"""Run yield projections for SSP3-7.0 using the existing v2 yield model.

Loads the SSP370 county climate projections and applies the trained LightGBM
yield model, saving results to data/projections/yield_projections_SSP370.parquet.
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
PROJECTIONS_DIR = PROJECT_ROOT / 'data' / 'projections'
RESULTS_DIR = PROJECT_ROOT / 'results'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

SCENARIO = 'SSP370'


def load_yield_model():
    """Load the most recent yield model from results directories.

    Returns:
        Trained LGBMRegressor.

    Raises:
        FileNotFoundError: If no yield model exists.
    """
    results_dirs = sorted(RESULTS_DIR.glob('20*'))
    for d in reversed(results_dirs):
        yield_path = d / 'yield_model.pkl'
        if yield_path.exists():
            with open(yield_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded yield model from {yield_path}")
            return model
    raise FileNotFoundError("No yield model found — run Phase 3 first")


def project_yields_ssp370(yield_model, climate_proj, panel):
    """Project county-crop yields under SSP3-7.0.
