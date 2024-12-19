"""
Compute bootstrap confidence intervals for the four headline findings.

Bootstrap unit: county (with replacement across counties), 1000 iterations, seed=42.
All dollar values in 2023 USD (inherited from upstream parquet files).
Saves results to state/confidence_intervals.json.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RNG = np.random.default_rng(42)
N_BOOT = 1000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
STATE = PROJECT_ROOT / "state"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_stat(values: np.ndarray, stat_fn, n_boot: int = N_BOOT, rng=RNG):
