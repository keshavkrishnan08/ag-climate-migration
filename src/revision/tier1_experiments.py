"""Tier-1 experiment battery: every substantive concern, run head-on.

E10: Market-efficiency rigorous multi-spec (resolve sign issue)
E11: Migration METRO-ONLY placebo (RUCC 1-3, never farm-dependent)
E12: Migration preferred-estimand declaration with effect-size dominance ratio
E13: Crop-specific operating margins for northern opportunity
E14: Yield z-scale per-crop honest breakdown (cotton/corn)
E15: Long-difference fiscal chain weighted
E16: DCF floor + indirect joint sensitivity grid
E17: Tail-risk decoupled (separate from main DCF)
E18: Hedonic SI section6 reconciliation ($187B vs $80B)
E19: Climate-projected income path for depop NPV (replaces uniform draw)
E20: Hedonic extended-controls battery
E21: Insurance climate-σ at multiple α (0.02, 0.04, 0.08)
E22: Insurance SCO + ECO joint contribution
E23: Discount-rate sensitivity at Giglio (2021) range
E24: Stranded model-average (ML/process/hedonic) Bayesian-style blend
E25: Migration wild-cluster bootstrap on YEAR dimension too
E26: Floor binding county count by pasture value
E27: Migration robustness to farm-intensity cutoff
E28: Insurance robustness to APH window 4/5/6/7/8/9/10
E29: Hedonic by region (south/midwest/plains)
E30: Common-cause exposure: alternate physical metric robustness

Seed 42; reads results/revision/*.json; writes results/revision/tier1_experiments.json.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
np.random.seed(42)
OUT = Path("results/revision")
out = {}

def jl(name):
    p = OUT / name
    return json.load(open(p)) if p.exists() else None
def gj(d, *path, default=None):
    cur = d
    for k in path:
        if cur is None: return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur if cur is not None else default

# ============================================================
# E10. MARKET-EFFICIENCY RIGOROUS MULTI-SPEC
# ============================================================
mr = jl("market_robustness.json") or {}
# What we have: multiple specs of cross-section Delta-lnV ~ Delta-T_2040
# Report ALL specs with effect size + interpretation
