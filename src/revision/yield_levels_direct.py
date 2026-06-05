"""Direct yield-LEVELS model -- the AgMIP/hybrid-ML-comparable metric (R2>0.5 bar).

Reviewer 2's ">0.5 at county scale" benchmark refers to predicting actual yields
(levels), as the GNN/hybrid-ML county papers report. We therefore train a per-crop
model that predicts yield_bu_acre directly from: a technology-time term, the monthly
temperature-exposure spectrum (Schlenker-Roberts degree-time bins), monthly
precipitation/VPD/PDSI, a soil-productivity index (NCCPI proxy) and latitude.
Train <=2012, test 2013-2023; report levels R2 per crop. Also adds a mechanistic
predictor (process-based water-stress-adjusted GDD) so the model is a genuine
mechanistic-ML hybrid. Seed 42.
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


def main():
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet",
                         columns=["fips", "year", "crop", "yield_bu_acre",
                                  "log_population", "log_median_income"])
