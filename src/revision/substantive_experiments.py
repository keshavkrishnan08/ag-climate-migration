"""Substantive experiments addressing the critique.

E1. Insurance falsification: process-based Schlenker-Roberts yield path (independent of ML)
E2. Insurance climate-dependent sigma: variance scales with projected temperature
E3. Migration metro placebo: clean falsification on metro-only counties
E4. Migration effect-size dominance: farm-dep beta / non-farm beta ratio
E6. Depopulation NATIONAL welfare floor: search-frictional cost only
E8. Stranded ML vs process explicit decomposition
E9. Stranded DCF without INDIRECT_MULTIPLIER (pure direct)

Seed 42; reads results/revision/ inputs; writes results/revision/substantive_experiments.json.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
np.random.seed(42)
OUT = Path("results/revision")
out = {}

# ============================================================
# E8 + E9: Stranded ML vs process + no-indirect-multiplier
# ============================================================
db = json.load(open(OUT / "dollar_robustness.json"))
ml = db["national_total_ml_B"]
proc = db["national_total_process_B"]
out["E8_stranded_ML_vs_process"] = {
    "ML_path_total_B": round(ml, 1),
    "process_path_total_B": round(proc, 1),
    "ratio_ML_over_process": round(ml / proc, 2),
