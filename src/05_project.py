"""Phase 4: Future projections — CMIP6 scenarios 2025-2050.

Projects county-level crop yields using:
1. Trained yield model (Phase 3A)
2. Crop switching models (Phase 3B)
3. CMIP6 climate projections downscaled to county level (pre-computed)

Primary scenario: SSP2-4.5 (~RCP 4.5), +1.4-1.8°C by 2050.
GCM ensemble: 5 CMIP6 models, median + 10-90th percentile uncertainty.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
import yaml
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
PROJECTIONS_DIR = PROJECT_ROOT / 'data' / 'projections'
RESULTS_DIR = PROJECT_ROOT / 'results'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

SCENARIOS = CONFIG['climate_scenarios']
RANDOM_SEED = CONFIG['yield_model']['random_seed']


def load_trained_models() -> dict:
    """Load yield model and switching models from Phase 3.

    Returns:
        Dict with 'yield_model' and 'switching_models' keys.

    Raises:
        FileNotFoundError: If model files don't exist.
    """
    results_dirs = sorted(RESULTS_DIR.glob('20*'))
    if not results_dirs:
        raise FileNotFoundError("No results directory found — run Phase 3 first")

    models = {}

    # Find yield model across results dirs
    for d in reversed(results_dirs):
        yield_path = d / 'yield_model.pkl'
        if yield_path.exists():
            with open(yield_path, 'rb') as f:
                models['yield_model'] = pickle.load(f)
            logger.info(f"Loaded yield model from {yield_path}")
            break

    switching_dir = RESULTS_DIR / 'switching_models'
    if switching_dir.exists():
        models['switching_models'] = {}
        for pkl_file in switching_dir.glob('*_model.pkl'):
            pair_name = pkl_file.stem.replace('_model', '')
            with open(pkl_file, 'rb') as f:
                models['switching_models'][pair_name] = pickle.load(f)
        logger.info(f"Loaded {len(models['switching_models'])} switching models")

    return models


def _f_to_c(f_val):
    """Convert Fahrenheit to Celsius."""
    return (f_val - 32) * 5 / 9


def project_yields(
    yield_model,
    climate_proj: pd.DataFrame,
    panel: pd.DataFrame,
    scenario: str
) -> pd.DataFrame:
    """Project county-crop yields under a climate scenario.

    Uses the trained LightGBM yield model on modified feature vectors.
    For each projection year:
      1. Start from most recent observed features per county-crop
      2. Apply climate deltas from CMIP6 projections
      3. Predict yield anomaly with the model
      4. Re-add extrapolated technology trend
