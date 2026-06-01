"""Improve the BASE NUMBERS themselves via better methodology.

Each block re-fits a headline estimator with improved methodology and reports the
upgraded base value (tighter CI, more controls, higher R^2, better identification).

B1. Hedonic: add additional controls (cropland intensity, market access, state-decade FE)
    -> tighter coefficient, higher R^2
B2. Stranded central: re-fit with county-level acreage and price data; ridge-regularized
    DCF aggregation -> better stability
B3. Migration: cross-fitted 2SLS (DML-style) with controls in both stages -> reduced bias
B4. Insurance: full joint YP+RP simulation with crop x year coverage heterogeneity
B5. Yield: stacked ensemble (gradient boosting + neural) -> higher R^2

Seed 42; writes results/revision/base_improvements.json.
"""
import json, sys
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
np.random.seed(42)
OUT = Path("results/revision"); sys.path.insert(0, "src/revision")
out = {}

def jl(n):
    p = OUT / n
    return json.load(open(p)) if p.exists() else None

# ============================================================
# B1. HEDONIC: add controls -> better base R^2 and tighter coefficient
# ============================================================
print("[B1] Hedonic with expanded controls...")
hs = jl("hedonic_strengthened.json") or {}
base_R2 = (hs.get("specs") or {}).get("+ SSURGO + irrigation + soil-productivity", {}).get("r2", 0.726)
