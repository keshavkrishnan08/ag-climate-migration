"""Tier-3: rigorously tighten every cited number.

Strategy:
- Migration wild-cluster bootstrap at B=9999 (vs 1999) -> tighter p-value precision
- Depopulation NPV at 1M draws (vs 200k) -> narrower 90% CI
- Stranded CI propagation at 1M draws
- Hedonic with HC3 robust SEs (more conservative than HC1) + cluster-robust on state
- Migration with HC3 + cluster bootstrap CIs
- Yield acreage-weighted Spearman (the valuation-relevant metric)
- Common-cause test with Bonferroni adjustment across 3 channels
- Insurance residual with cluster-robust over crop x year + 95% interval
- All robustness checks done at higher precision

Every output is paired with a STATISTICAL RIGOR delta showing the improvement vs the
prior result (tighter CI, lower p, etc.).

Seed 42; writes results/revision/tier3_tighten.json.
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
