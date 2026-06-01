"""E51: Compete the single-factor SEM against two alternative structures.

Reviewer concern: with 4 channels and 2 degrees of freedom, single-factor fit
is nearly guaranteed; it cannot distinguish a genuine common institutional
cause from shared exposure to a thermal driver (e.g., July Tmax). We compete:

  M0 (saturated)        : correlated 4-channel covariance, df = 0
  M1 (single factor)    : 4 loadings, df = 2 (the baseline reported)
  M2 (two-factor)       : 2 latents = {institutional pricing} and {direct
                          thermal exposure}, with C1, C2 loading on F1 and
                          C3, C4 loading on F2 (and an inter-factor
                          correlation phi); df = 1
  M3 (correlated errors): single-factor + a correlated residual between C1
                          (stranded) and C2 (insurance), the two channels
                          that share the EDD signal; df = 1

We compare on chi-square difference, AIC, and the Vuong test on the
loglikelihood ratio. The single-factor model wins if (i) the two-factor and
correlated-error AIC values do not improve on M1 by Delta_AIC > 2 and
(ii) the chi-square decrement is not significant. We then run a covariate-
adjusted single-factor SEM that partials out July Tmax exposure at the
county level; if loadings shrink toward zero, the institutional reading is
not separable from shared exposure. Seed 42.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist

OUT = Path("results/revision")
rng = np.random.default_rng(42)
n = 1820

# Observed inter-channel partial correlation matrix from common_cause_sem.py.
obs = np.array([
    [1.00, 0.41, 0.39, 0.36],
    [0.41, 1.00, 0.44, 0.37],
    [0.39, 0.44, 1.00, 0.42],
    [0.36, 0.37, 0.42, 1.00],
