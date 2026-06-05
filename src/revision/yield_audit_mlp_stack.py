"""Audit improvement B: MLP on the monthly climate sequence + stacked ensemble.

Orthogonal architecture to the gradient-boosted trees. Two distinct ideas:

1. MLP over the raw MONTHLY climate sequence. A tree splits one feature at a time
   and approximates smooth multivariate climate response with axis-aligned steps.
   A small multi-layer perceptron fed the standardized monthly tmax/tmin/precip/PDSI
   vector (Apr-Sep) plus soil/latitude/trend can capture the smooth, interacting
   temperature x water response directly. We keep CLIMATE-ONLY inputs so it stays
   projectable from CMIP6 monthly deltas (no spatial-yield lags).

2. STACK the MLP with the best gradient-boosted tree (v4-style feature set + the
   drought-trajectory features) via a non-negative least-squares blend fit on a
   held-out 2010-2012 window (so blend weights never see the 2013-2023 test set).
   If the two learners make different errors, the stack beats either alone.

Target = the existing z-scored, county-detrended yield anomaly, so the held-out
R^2 is DIRECTLY comparable to the paper's 0.227. Split: train<=2009 (MLP needs a
clean fit window), blend 2010-2012, test 2013-2023. Seed 42.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from scipy.optimize import nnls
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_audit_drought_huber import build_panel, design

DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "revision"
SEED = 42
GROW = [f"{m:02d}" for m in range(4, 10)]


def monthly_sequence_features():
    """Standardizable monthly tmax/tmin/precip/PDSI for Apr-Sep, plus county means.

    Returns:
        DataFrame [fips, year, <monthly climate columns>].
    """
    m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet")
    m["fips"] = m["fips"].astype(str).str.zfill(5)
