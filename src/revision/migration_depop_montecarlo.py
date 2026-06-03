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

