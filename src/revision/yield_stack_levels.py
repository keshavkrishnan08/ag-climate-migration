"""Stacked levels ensemble: gradient-boosted trees + MLP (NNLS blend), per crop,
on cached features incl. irrigation + lagged yield. The hybrid-network architecture
the benchmark papers use. Reports levels R2 per crop. Seed 42."""
import json, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from scipy import stats
from scipy.optimize import nnls
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"results"/"revision"
