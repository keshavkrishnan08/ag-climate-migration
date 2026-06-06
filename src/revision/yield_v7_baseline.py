"""Fair comparison: same %-deviation target, OLD growing-season aggregate features
(no temperature spectrum). Isolates how much the spectrum representation adds vs
the original aggregates, holding the target fixed.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_v5_percrop import latitude
DATA_PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "revision"
SEED = 42


def main():
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet")
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
