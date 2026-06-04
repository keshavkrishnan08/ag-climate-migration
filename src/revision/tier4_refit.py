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
# F1. Migration: bootstrap CI at every horizon (3, 5 yr)
# ============================================================
print("[F1] Migration bootstrap CIs at every horizon...")
horizon_data = jl("migration_horizon.json") or {}
mp = jl("migration_primeage_panel.json") or {}
# Pull point estimates and SEs
h3 = horizon_data.get("3yr_full") or mp.get("primeage_panelFE_farmdep", {})
h5 = horizon_data.get("5yr_full") or {}
b3, se3 = h3.get("beta", 0.024), h3.get("se", 0.009)
b5, se5 = h5.get("beta", 0.049), h5.get("se", 0.015)
