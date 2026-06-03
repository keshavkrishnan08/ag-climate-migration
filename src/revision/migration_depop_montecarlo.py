"""Replace the ad-hoc $6-39B depopulation-cost range with a propagated Monte Carlo CI.
The depopulation NPV is the foregone regional output of the climate-driven prime-age
displacement (and their households) in farming-dependent counties, accumulated to 2050 and
discounted. Rather than hand-set scenario bounds, we draw every input from its documented
range -- crucially including the IV elasticity's own sampling distribution -- and report the
resulting central estimate and 5-95% interval. Seed 42; writes only to results/revision/.

Chain (per draw):
  annual prime-age shortfall at 2050  D = beta * income_decline * prime_age_base
  total persons displaced              N = D * household_factor
  full-displacement annual output      Y = N * per_capita_output * local_multiplier
  NPV  = sum_{t=1..H} (t/H) * Y / (1+r)^t      [linear phase-in 2024->2050, H=26 yr]
"""
import numpy as np, json
from pathlib import Path
OUT = Path("results/revision")
rng = np.random.default_rng(42)

PRIME_AGE_BASE = 1_130_330      # farming-dependent counties, 2019 (results/revision/migration_aggregate.json)
H = 26                          # 2024 -> 2050
N = 200_000

# --- parameter draws over documented ranges ---
# IV 5-yr elasticity: estimated N(0.0491, 0.0149); truncate at >0 (the identified sign)
beta = rng.normal(0.0491, 0.0149, N); beta = np.clip(beta, 0, None)
income_decline = rng.uniform(0.15, 0.25, N)     # SSP2-4.5 -> SSP3-7.0 sustained farm-income decline
household      = rng.uniform(2.2, 2.6, N)        # total residents per displaced prime-age worker (doc 2.2-2.4; empirical median 3.09)
per_capita     = rng.uniform(70_000, 75_000, N)  # regional per-capita output, 2023 USD
multiplier     = rng.uniform(1.6, 1.8, N)        # USDA-ERS / IMPLAN rural farm-economy output multiplier
disc           = rng.uniform(0.03, 0.05, N)      # discount rate

D = beta * income_decline * PRIME_AGE_BASE       # annual prime-age shortfall at full displacement
persons = D * household
annual_output = persons * per_capita * multiplier

t = np.arange(1, H + 1)
# NPV with linear phase-in: vectorize over draws
npv = np.array([(annual_output[i] * (t / H) / (1 + disc[i]) ** t).sum() for i in range(N)])
npv_B = npv / 1e9

# workers-only floor (no household scaling, personal-income basis ~ per-capita * 1.0 multiplier)
annual_floor = D * per_capita        # workers only, no multiplier
npv_floor = np.array([(annual_floor[i] * (t / H) / (1 + disc[i]) ** t).sum() for i in range(N)]) / 1e9

