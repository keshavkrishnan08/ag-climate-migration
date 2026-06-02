"""E53: Migration elasticity attenuation under regime change.

Reviewer concern: the 90% MC interval [$11, $38]B captures sampling
uncertainty in beta but not regime-change uncertainty. We test a beta-
attenuation sensitivity: beta(t) starts at the estimated 0.049 in 2024 and
declines linearly to a target by 2040 (and stays there to 2050). Three
attenuation paths:

  No attenuation  : beta(t) = 0.049 (baseline; matches the published headline)
  Modest          : beta(t) -> 0.0735 (1.5x) by 2040, then flat
