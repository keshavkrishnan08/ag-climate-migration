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
])
iu = np.triu_indices(4, 1)


def ssr_to_chi2(ssr):
    return (n - 1) * ssr


def fit_stats(implied, free_params):
    diff = obs - implied
    ssr = float(np.sum(diff[iu] ** 2))
    chi2 = ssr_to_chi2(ssr)
    df = 6 - free_params
    null_chi2 = (n - 1) * float(np.sum(obs[iu] ** 2))
    cfi = max(0.0, 1 - max(chi2 - df, 0) / max(null_chi2 - 6, 1))
    rmsea = float(np.sqrt(max((chi2 / max(df, 1) - 1) / (n - 1), 0)))
    srmr = float(np.sqrt(np.mean(diff[iu] ** 2)))
    p = float(1 - chi2_dist.cdf(chi2, df)) if df > 0 else 1.0
    aic = chi2 - 2 * df  # standard SEM AIC = chi2 + 2k_free - 2*df_baseline (relative)
    return {
        "chi_square": round(chi2, 3),
        "df": int(df),
        "p_chi_square": round(p, 3),
        "CFI": round(cfi, 3),
        "RMSEA": round(rmsea, 3),
        "SRMR": round(srmr, 3),
        "AIC": round(aic, 2),
    }


# M1 single factor
def m1_obj(L):
    imp = np.outer(L, L)
    np.fill_diagonal(imp, 1.0)
    return float(np.sum((obs[iu] - imp[iu]) ** 2))


L1 = minimize(m1_obj, np.full(4, 0.6), method="L-BFGS-B",
              bounds=[(0.1, 0.99)] * 4).x
imp1 = np.outer(L1, L1)
