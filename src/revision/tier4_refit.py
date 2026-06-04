"""Tier-4: deepest rigor. Actually compute HC3 robust SEs for hedonic, bootstrap CIs at
every migration horizon, narrow Monte Carlo intervals further with antithetic variates,
add Romano-Wolf multiple-testing across migration specs, and produce final CI table.

Seed 42; writes results/revision/tier4_refit.json.
"""
import json, sys
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
np.random.seed(42)
OUT = Path("results/revision")
sys.path.insert(0, "src/revision")
out = {}

def jl(n):
    p = OUT / n
    return json.load(open(p)) if p.exists() else None

# ============================================================
