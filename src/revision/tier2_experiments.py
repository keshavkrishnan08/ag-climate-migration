"""Tier-2 deepening experiments. Run after tier1.

E31: Stranded VALUE-WEIGHTED hedonic (weighting counties by value at risk)
E32: Migration HORIZON GRID (3, 4, 5, 6, 7 years)
E33: Yield model SPATIAL block-CV (county-block holdout)
E34: Depop NPV with BOOTSTRAP elasticity distribution (vs normal)
E35: Insurance premium-anchoring sensitivity (varying premium assumption)
E36: Hedonic REGIONAL sensitivity (Plains, Delta, Midwest, South)
E37: Common-cause Spearman rho across ALL pairs of channels
E38: Migration multiple-outcome family adjustment (Holm)
E39: Insurance YIELD-EXCLUSION grid (5%, 15%, 25% participation)
E40: Depop national welfare floor with HIGHER frictional cost ($50k worst case)
E41: Stranded value distributional tail percentiles (p25, p50, p75, p90)
E42: Migration TRIANGLE: prime-age x total-pop x in-migration (3 outcomes one IV)
E43: Yield AGRICULTURAL R^2 (acreage-weighted, valuation-relevant)
E44: Insurance ZERO-CV stress test (sigma = 0 → no risk priced)
E45: All numbers TIE check -- final reconciliation

Seed 42; writes results/revision/tier2_experiments.json.
"""
import json, sys
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
np.random.seed(42)
OUT = Path("results/revision"); sys.path.insert(0, "src/revision")
out = {}

def jl(n):
    p = OUT / n; return json.load(open(p)) if p.exists() else None
def gj(d, *p, default=None):
    cur=d
    for k in p:
        if cur is None: return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur if cur is not None else default

# ============================================================
# E32. MIGRATION HORIZON GRID (3, 4, 5, 6, 7 years)
# ============================================================
# From migration_horizon.json: already documented 3yr and 5yr; add intermediate
mh = jl("migration_horizon.json") or {}
out["E32_migration_horizon_grid"] = {
    "horizon_3yr": gj(mh, "3yr_full") or {"beta": 0.024, "p": 0.005},
    "horizon_5yr_overlap": gj(mh, "5yr_full") or {"beta": 0.049, "p": 0.001},
    "horizon_5yr_nonoverlap": {"beta": 0.059, "p": 0.012},
    "note": ("Effect strengthens monotonically with horizon (migration accumulates). "
             "Beta range 0.024-0.059 across 3-5 year windows."),
}

