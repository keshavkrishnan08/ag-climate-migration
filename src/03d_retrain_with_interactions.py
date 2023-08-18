"""Phase 3D: Retrain yield model with compound drought interaction features.

Reviewer fix: the base model under-predicts the 2012 drought because it misses
the multiplicative penalty when heat AND moisture stress occur simultaneously.
This script adds compound interaction features and retrains with depth=8 to
let the tree structure capture the threshold behavior.

New features added:
    heat_x_drought     = tmax_july_c_anomaly * (-pdsi_growing_anomaly)
    heat_x_precip      = tmax_july_c_anomaly * (-precip_growing_anomaly)
    extreme_compound   = (tmax_anom > 1°C) AND (pdsi_anom < -1) binary flag
    tmax_july_sq       = tmax_july_c_anomaly² (quadratic heat threshold)
    precip_deficit     = max(0, county_baseline_precip - actual_precip)
    tmax_peak_c        = max(tmax Jun/Jul/Aug) in Celsius (from monthly data)
    precip_jja         = total Jun+Jul+Aug precipitation
    pdsi_peak_drought  = min(PDSI Jun/Jul/Aug) — worst summer drought month
    edd_months_c       = months where tmax > 33.5°C (Schlenker & Roberts threshold)
    edd_x_pdsi         = edd_months_c * (-pdsi_peak_drought) compound signal
    + anomaly versions of the monthly features (county mean subtracted)

Temporal split (matches config.yaml val_end=2012):
    Train:  years <= 2012
    Test:   2013-2023

Saves:
    results/yield_model_v2.pkl   — new primary model
    results/yield_model_v2_metrics.json
    results/feature_importance_v2.csv
"""

import os
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
