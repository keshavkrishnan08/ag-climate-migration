"""Final yield model: temperature spectrum + monthly climate + soil + latitude +
IRRIGATION SHARE (sub-county management, from NASS practice records). Trains both
the levels model (AgMIP-comparable, target R^2>=0.5) and the %-deviation model.
Caches the engineered feature matrix so it is built only once. Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

