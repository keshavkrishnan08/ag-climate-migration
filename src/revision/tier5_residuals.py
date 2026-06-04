"""Tier-5: close residual critique items via experiments.

R1. Insurance PARTIAL-EQUILIBRIUM premium anchoring -> re-run with rate-adjustment over time
R2. Migration TREATMENT/INSTRUMENT alignment -> reconstruct treatment to match instrument's
    yield-shock content (drop national price variation from treatment)
R3. Infrastructure $36B engineering validation -> compare to per-bushel grain storage cost
    benchmarks (USDA, NEPC capital cost surveys)
R4. AgMIP yield apples-to-apples -> R^2 on the AgMIP target (yield levels, growing-season
    aggregates, no remote sensing) at the same county scale
R5. Migration with bootstrap CIs at every horizon AND every farm-intensity cut
R6. Stranded final sensitivity grid (discount x horizon x price x floor x indirect, 5^5)
R7. Yield model held-out SPATIAL block CV (climate-region blocks)
R8. Depop NPV with full empirical income path (from per-county SSP yield -> revenue)
R9. Insurance with NET indemnity payments accounting (not just expected)
R10. Common-cause IRT (item-response model) -> latent factor share

Seed 42; writes results/revision/tier5_residuals.json.
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
# R1. INSURANCE: partial-eq vs general-eq premium anchoring
# ============================================================
print("[R1] Insurance with rate adjustment over time...")
# RMA adjusts premium RATES (per dollar liability) over time as experience updates.
# The headline residual treats premium per acre as anchored at observed; this is
# partial-equilibrium. General-equilibrium would re-rate as APH lags persist.
#
# Approximation: RMA loss-ratio targets are 1.0; if true expected loss ratio rises
# from observed (~0.65 historical) toward 1.0+ under non-stationary climate, premiums
# would mechanically rise as well. The mispricing residual converges back to zero in
# the very long run as rates adjust.
#
# Time-to-adjustment is the key parameter. RMA premium rates lag observed losses with
# a 5-year averaging window. So the rate-adjusted residual decays as exp(-t/5):
#
# Mispricing_GE(t) = Mispricing_PE * exp(-t/tau_rate)
#
# At tau_rate = 5 years, the residual halves in ~3.5 years.
# At infinity, GE residual = 0 (full re-rating).
