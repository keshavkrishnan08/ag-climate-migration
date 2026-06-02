"""Dollar-robustness to yield-model specification (Reviewer 2's explicit alternative:
"convince me that a more sophisticated specification leaves the dollar conclusions
unchanged").

We compute county-level conservative stranded value under two very different yield
specifications -- the statistical ML model and a purely process-based Schlenker-
Roberts extreme-degree-day damage function -- and show (a) the national total stays
the same order of magnitude, (b) the spatial pattern (which counties are exposed)
is highly rank-correlated, and (c) the hedonic upper bound ($168B) uses no yield
model at all. The policy conclusion is therefore not load-bearing on the yield model.
Seed 42.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
PROJ = ROOT / "data" / "projections"
OUT = ROOT / "results" / "revision"
rp = pd.read_csv(OUT / "real_prices_2023usd.csv"); PRICE = dict(zip(rp.iloc[:, 0], rp.iloc[:, 1]))
SR = {"corn": -0.0662, "soybeans": -0.0560, "wheat_winter": -0.0420, "wheat_spring": -0.0420,
      "cotton": -0.0662, "sorghum": -0.0662, "barley": -0.0420, "oats": -0.0420}


def edd(tj, tg, thr=29.0):
    return np.maximum(0, tj - thr) * 31 + np.maximum(0, tg - thr) * 60


def county_stranded_ml(scen="SSP245", r=0.04, H=30):
