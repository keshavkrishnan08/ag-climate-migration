"""
Fix 4: IV estimation of farm income -> outmigration elasticity.

Approach: Two-stage least squares with two-way county + year FE.

Instrument: Weather-driven income shock. For each county-year, we compute:
    Z_it = sum_c [ yield_detrended_ict * acres_ic_bar * price_c ] / baseline_income_i
    where yield_detrended = actual yield minus county-crop quadratic trend,
    acres_ic_bar = county-crop mean acres (fixed exposure),
    price_c = national commodity price.
    This isolates the weather-driven component of farm revenue.

Treatment: Farm income deviation from county baseline.
    D_it = (income_it - income_i_bar) / income_i_bar
    where income_it = sum_c [ yield_ict * acres_ict * price_c * deflator ]

Outcome: Net outmigration rate from population change.
    Y_it = -(pop_t - pop_{t-1}) / pop_{t-1}
    Positive = population loss = net outmigration.

Sample: Rural Corn Belt counties (pop < 50,000), 2010-2023 (ACS range).
FE:     County + year (absorbed via two-way demeaning).
SE:     Cluster-robust at county level.
Gate:   First-stage F > 10.

Uses manual 2SLS via numpy/scipy (statsmodels has scipy compat issue).

Author: Keshav Krishnan
Date:   2026-03-17
"""
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
CORN_BELT_STATE_FIPS = [
    "19",  # Iowa
    "17",  # Illinois
    "18",  # Indiana
    "39",  # Ohio
    "27",  # Minnesota
    "55",  # Wisconsin
    "31",  # Nebraska
    "29",  # Missouri
    "46",  # South Dakota
    "38",  # North Dakota
    "20",  # Kansas
]

POP_CAP = 50_000  # rural threshold

# Commodity prices in 2023 USD (5-year average, $/bushel except cotton $/lb)
COMMODITY_PRICES = {
    "corn": 5.50,
    "soybeans": 12.80,
    "wheat_winter": 7.20,
    "wheat_spring": 8.10,
    "cotton": 0.78,      # $/lb
    "sorghum": 5.30,
    "barley": 6.10,
    "oats": 3.80,
}

CPI_2023 = 304.703  # from data/raw/other/cpi_annual.csv

RANDOM_SEED = 42

# Min yield thresholds to filter silage/cover crop records
MIN_YIELD = {
    "corn": 50,
    "soybeans": 10,
    "wheat_winter": 10,
    "wheat_spring": 10,
    "sorghum": 15,
    "barley": 10,
    "oats": 10,
    "cotton": 100,  # lbs/acre
}


def ols_fit(y, X):
    """Fit OLS via numpy least squares.

    Args:
        y: Response vector (n,).
        X: Design matrix (n, k). Should include constant if desired.

    Returns:
        Dict with coefficients, residuals, fitted values, and diagnostics.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n, k = X.shape

    beta, _, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    fitted = X @ beta
    resid = y - fitted
    ss_res = resid @ resid
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    sigma2 = ss_res / (n - k)
    var_beta = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(var_beta))
    t_stats = beta / se

