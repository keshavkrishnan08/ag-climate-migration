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
specs = {}
for k, v in mr.items():
    if isinstance(v, dict) and "beta" in v and "p" in v:
        specs[k] = {"beta": round(v["beta"], 4), "p": round(v["p"], 4), "n": v.get("n")}
out["E10_market_efficiency_multispec"] = {
    "specifications": specs,
    "interpretation": ("Across specifications: coefficient sign and significance are NOT stable. "
                       "Cross-sectional Δlog(V) ~ ΔT(2040) cannot reject 'no forward discounting' "
                       "but also does not provide active confirmation of unpriced risk -- the "
                       "market test is INDETERMINATE and is reported as such."),
    "n_specs_significant_negative": sum(1 for s in specs.values() if s["beta"] < 0 and s["p"] < 0.05),
    "n_specs_significant_positive": sum(1 for s in specs.values() if s["beta"] > 0 and s["p"] < 0.05),
    "n_specs_null": sum(1 for s in specs.values() if s["p"] >= 0.05),
}

# ============================================================
# E11. METRO-ONLY MIGRATION PLACEBO (truly non-farm)
# ============================================================
# Restrict to metro counties (high population density, never farm-dependent).
# If the Bartik instrument moves population there, exclusion is violated.
import sys; sys.path.insert(0, "src/revision")
try:
    from migration_iv_bartik import build_panel
    panel = build_panel()
    panel["fips"] = panel["fips"].astype(str).str.zfill(5)
    # Metro = high population, low farm dependency
    # Use total_population >= 100k AND not farm_dependent
    high_pop_fips = panel[panel.year == 2019].groupby("fips")["total_population"].max()
    high_pop_fips = high_pop_fips[high_pop_fips > 100_000].index
    metro = panel[(panel.fips.isin(high_pop_fips)) & (panel.farm_dependent == 0)].copy()
    metro = metro.dropna(subset=["pop_growth_3yr", "z_bartik"])
    # Reduced form: pop growth ~ z_bartik with county+year FE
    metro["pgw"] = metro["pop_growth_3yr"] - metro.groupby("fips")["pop_growth_3yr"].transform("mean") \
                   - metro.groupby("year")["pop_growth_3yr"].transform("mean") + metro["pop_growth_3yr"].mean()
    metro["zw"] = metro["z_bartik"] - metro.groupby("fips")["z_bartik"].transform("mean") \
                  - metro.groupby("year")["z_bartik"].transform("mean") + metro["z_bartik"].mean()
    X = np.column_stack([np.ones(len(metro)), metro["zw"].values])
    Y = metro["pgw"].values
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r = Y - X @ b
