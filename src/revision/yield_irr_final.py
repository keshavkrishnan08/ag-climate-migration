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


def build_features():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet",
                         columns=["fips", "year", "crop", "yield_bu_acre",
                                  "log_population", "log_median_income", "yield_trend_slope_15yr"])
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    fm = fm[fm["yield_bu_acre"] > 0].copy()
    m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet")
