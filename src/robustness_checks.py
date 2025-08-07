"""Robustness checks for Nature Food reviewers.

Six checks addressing the top reviewer concerns:
  1. Hedonic with soil quality proxy (historical yield 1990-2005 as soil proxy)
  2. Leave-one-crop-out sensitivity for stranded asset computation
  3. Leave-one-GCM-out sensitivity for stranded asset computation
  4. Placebo test: run cascade on LEAST climate-affected counties (top quartile positive impact)
  5. Temporal stability of the hedonic regression (2010-2015 vs 2015-2022)
  6. Insurance mispricing under alternative coverage levels (65% and 85%)

Each check saves results to results/robustness/ and prints a one-line verdict:
  ROBUST   — result is insensitive to the specification change
  SENSITIVE — result changes materially; report both
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from loguru import logger
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROJ    = PROJECT_ROOT / "data" / "projections"
RESULTS_DIR  = PROJECT_ROOT / "results" / "robustness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

CPI_2023 = 304.7
CPI_2022 = 296.8
DEFLATOR_2022 = CPI_2023 / CPI_2022
SEED = 42
np.random.seed(SEED)

GROWING_MONTHS = [4, 5, 6, 7, 8, 9]

COMMODITY_PRICES = {
    "corn": 5.50, "soybeans": 12.80, "wheat_winter": 7.20,
    "wheat_spring": 8.10, "cotton": 0.78, "sorghum": 5.30,
    "barley": 6.10, "oats": 3.80,
}

# From the main hedonic run — reference coefficients for stability comparison
BASELINE_BETA_TMAX     = None   # filled in at runtime from check 1 baseline run
BASELINE_STRANDED_B    = 168.0  # hedonic $168B from headline_numbers_preliminary.json
DCF_CONSERVATIVE_B     = 76.0   # DCF conservative from same file

VERDICTS: list[dict] = []


# ──────────────────────────────────────────────────────────────────────────────
