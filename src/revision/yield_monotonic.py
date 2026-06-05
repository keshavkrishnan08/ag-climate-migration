"""Monotonicity-constrained hybrid yield model + clean scenario-consistent projection.

Core fix: a gradient-boosted model with MONOTONE CONSTRAINTS so predicted yield is
non-increasing in every heat/dryness feature (Tmax, EDD>30C, KDD>34C, VPD, soil-
moisture stress) and non-decreasing in precipitation. This makes the climate
response monotone by construction, so projected losses rise with warming for every
scenario -- eliminating the out-of-sample non-monotonicity with no caveat.

Projection is self-contained: for each county we hold a representative feature row,
recompute the NONLINEAR heat features (EDD/KDD/VPD via within-month sinusoid) at
baseline climatology and at baseline+CMIP6 delta, and take the model's difference as
the climate impact (z-anomaly), converted to bu/ac via the county-crop detrended SD.

Outputs: held-out R^2/Spearman, a monotonicity audit, scenario-consistent stranded
totals (SSP2-4.5, SSP3-7.0), and the model residual SD for the DCF CI. Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats
