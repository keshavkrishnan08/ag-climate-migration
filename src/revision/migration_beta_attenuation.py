"""E53: Migration elasticity attenuation under regime change.

Reviewer concern: the 90% MC interval [$11, $38]B captures sampling
uncertainty in beta but not regime-change uncertainty. We test a beta-
attenuation sensitivity: beta(t) starts at the estimated 0.049 in 2024 and
declines linearly to a target by 2040 (and stays there to 2050). Three
attenuation paths:

  No attenuation  : beta(t) = 0.049 (baseline; matches the published headline)
  Modest          : beta(t) -> 0.0735 (1.5x) by 2040, then flat
  Halving         : beta(t) -> 0.0245 (0.5x) by 2040, then flat
  Quartering      : beta(t) -> 0.0123 (0.25x) by 2040, then flat

For each path, we rebuild the present-value depopulation NPV by replicating
the Monte Carlo headline procedure (200,000 draws, antithetic variates) but
with a time-varying beta multiplier on each year's contribution to displaced
prime-age population. Seed 42.
"""
import json
from pathlib import Path
