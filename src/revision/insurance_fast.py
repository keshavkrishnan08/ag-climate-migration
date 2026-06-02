"""Vectorized insurance mispricing simulation + round-2 robustness battery.

Reimplements the rolling-APH decomposition (insurance_rolling_aph.py) without the
per-county Python loop, so we can sweep robustness dimensions a tough reviewer
would request next:
  * APH window length (4 / 7 / 10 years) -- shorter windows absorb the trend
    faster, reducing the residual.
  * Yield-Exclusion (YE) at participation -- drop the worst year in the window,
    raising APH in disaster-prone counties (works against TAY).
  * Climate scenario (SSP2-4.5 vs SSP3-7.0).
  * Per-crop decomposition.

A vectorized re-implementation also cross-checks the headline produced by the
slow loop. Seed 42; writes only to results/revision/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from insurance_rolling_aph import (build_rma_county_crop, PRICE, LOADING, MAX_RATIO,
                                    TAY_PARTICIPATION, TAY_LAG_YEARS)

DATA_PROCESSED = ROOT / "data" / "processed"
PROJ = ROOT / "data" / "projections"
OUT = ROOT / "results" / "revision"
np.random.seed(42)
WIN = (2040, 2050)


def expected_indemnity_vec(K, mu, sigma):
    sigma = np.maximum(sigma, 1.0)
    z = (K - mu) / sigma
    return np.maximum((K - mu) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z), 0.0)
