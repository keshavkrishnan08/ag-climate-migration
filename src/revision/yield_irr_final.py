"""Final yield model: temperature spectrum + monthly climate + soil + latitude +
IRRIGATION SHARE (sub-county management, from NASS practice records). Trains both
the levels model (AgMIP-comparable, target R^2>=0.5) and the %-deviation model.
Caches the engineered feature matrix so it is built only once. Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_v7_spectrum import temperature_spectrum, es, GROW
from yield_v5_percrop import latitude
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "revision"
SEED = 42
CACHE = OUT / "yield_features_full.parquet"

