"""Revision: yield model with modern agro-climatic features (Reviewer 2 #1).

Reviewer 2 argued the yield model is under-specified relative to current
practice: it omits vapour pressure deficit (VPD), explicit heat-stress /
extreme-degree-day exposure, and soil-moisture stress -- all standard in the
modern crop-yield literature. This script engineers those features from the
monthly nClimDiv panel (tmax/tmin/precip/PDSI) and retrains the same three-model
ensemble (LightGBM + Ridge + RandomForest, NNLS blend) on the SAME temporal
split (train <= 2012, held-out test 2013-2023), so the improvement is
attributable to the features, not to leakage.

New features (growing season = months 04-09):
  vpd_growing      : mean vapour pressure deficit (kPa), FAO-56 with tmin as the
                     dewpoint proxy (ea = es(tmin), VPD = es(tmax) - es(tmin)).
  vpd_july         : July VPD (peak-demand month).
  edd30_growing    : extreme degree-days above 30 C, approximated month-by-month
                     from monthly tmax via a within-month sinusoid.
  heat_days_proxy  : count of growing-season months with tmax > 30 C.
  sm_stress        : soil-moisture stress = -min(growing-season PDSI) (driest
                     month deficit; larger = drier).
  sm_stress_july   : -PDSI in July.
  vpd_x_sm         : VPD x soil-moisture stress interaction (compound heat-drought).

All anomaly-standardised against the county mean so the model predicts the
z-scored yield anomaly (the existing target). Seed 42.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

