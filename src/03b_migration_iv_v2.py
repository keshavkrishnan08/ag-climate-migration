"""
Fix 4 (v2): IV estimation — farm income -> outmigration elasticity.
Three new specifications targeting cleaner out-migration measurement.

Approach 1: Net migration from population components
    net_out_migration_rate = -(pop_t - pop_{t-1}) / pop_{t-1}
    BUT cleaned:
      - Remove counties with |pop_change| > 10% (boundary changes)
      - 3-year rolling population change to smooth noise
      - Weighted by baseline population (upweight large counties)

Approach 2: Same-house inverse proxy
    mobility_rate = 1 - (same_house / total_population)
    = (mobility_total - same_house) / total_population
    Interpretation: fraction of current residents who moved into the county
    in the past year. Closely parallels Spec C/B but includes same-county movers.
    Expected sign: positive (income -> more in-migration -> higher mobility rate).
    Note: this is an IN-migration proxy, not pure out-migration, but captures
    the same economic channel. Lower income -> fewer arrivals AND more departures.

Approach 3: Gross out-migration via population accounting
    gross_out_rate = net_outmig_rate + in_migration_rate
    = -(pop_t - pop_{t-1})/pop_{t-1} + true_diff_county_in_rate
    Captures actual outflows regardless of inflow variation.

Summary of all IV specifications:
    Spec A  : net outmigration (raw pop change)              [original, p=0.49]
    Spec A2 : net outmigration cleaned (excl boundary, 3yr)  [Approach 1]
    Spec A3 : net outmigration pop-weighted                   [Approach 1 variant]
    Spec B  : gross mobility rate (1 - same_house/pop)        [Approach 2]
    Spec C  : true diff-county in-migration (B07001_049E)     [prior primary fix, p=0.019]
    Spec D  : long-distance in-migration (diff-county + diff-state)
    Spec E  : gross out-migration (net + in-migration)        [Approach 3]

Author: Keshav Krishnan
Date:   2026-03-18
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

POP_CAP = 50_000          # rural threshold
POP_CHANGE_EXTREME = 0.10 # exclude counties with |pop_change| > 10% (boundary changes)

# Commodity prices in 2023 USD (5-year average)
COMMODITY_PRICES = {
    "corn": 5.50,
    "soybeans": 12.80,
    "wheat_winter": 7.20,
    "wheat_spring": 8.10,
    "cotton": 0.78,
    "sorghum": 5.30,
    "barley": 6.10,
    "oats": 3.80,
}

CPI_2023 = 304.703

# Min yield thresholds to filter silage/cover crop records
MIN_YIELD = {
    "corn": 50,
    "soybeans": 10,
    "wheat_winter": 10,
    "wheat_spring": 10,
    "sorghum": 15,
    "barley": 10,
    "oats": 10,
    "cotton": 100,
}


# ──────────────────────────────────────────────────────────────────────
