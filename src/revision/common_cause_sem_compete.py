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
np.fill_diagonal(imp1, 1.0)
M1 = fit_stats(imp1, free_params=4)
M1["loadings"] = [round(float(v), 3) for v in L1]


# M2 two-factor: F1 -> C1,C2; F2 -> C3,C4; correlation phi.
def m2_obj(p):
    l1, l2, l3, l4, phi = p
    F = np.array([
        [1, phi],
        [phi, 1],
    ])
    Lam = np.array([
        [l1, 0],
        [l2, 0],
        [0, l3],
        [0, l4],
    ])
    imp = Lam @ F @ Lam.T
    np.fill_diagonal(imp, 1.0)
    return float(np.sum((obs[iu] - imp[iu]) ** 2))


p2 = minimize(m2_obj, [0.6, 0.6, 0.6, 0.6, 0.5],
              method="L-BFGS-B",
              bounds=[(0.1, 0.99)] * 4 + [(-0.99, 0.99)]).x
l1, l2, l3, l4, phi = p2
F = np.array([[1, phi], [phi, 1]])
Lam = np.array([[l1, 0], [l2, 0], [0, l3], [0, l4]])
imp2 = Lam @ F @ Lam.T
np.fill_diagonal(imp2, 1.0)
M2 = fit_stats(imp2, free_params=5)
M2["loadings_F1"] = [round(float(l1), 3), round(float(l2), 3)]
M2["loadings_F2"] = [round(float(l3), 3), round(float(l4), 3)]
M2["phi"] = round(float(phi), 3)


# M3 correlated-error single-factor + theta_{C1,C2}
def m3_obj(p):
    L = p[:4]
    theta = p[4]
    imp = np.outer(L, L)
    np.fill_diagonal(imp, 1.0)
    imp[0, 1] += theta
    imp[1, 0] += theta
    return float(np.sum((obs[iu] - imp[iu]) ** 2))


p3 = minimize(m3_obj, np.append(np.full(4, 0.6), 0.0),
              method="L-BFGS-B",
              bounds=[(0.1, 0.99)] * 4 + [(-0.3, 0.3)]).x
L3 = p3[:4]
theta = p3[4]
imp3 = np.outer(L3, L3)
np.fill_diagonal(imp3, 1.0)
imp3[0, 1] += theta
imp3[1, 0] += theta
M3 = fit_stats(imp3, free_params=5)
M3["loadings"] = [round(float(v), 3) for v in L3]
M3["theta_C1_C2"] = round(float(theta), 3)


# Likelihood-ratio tests M1 vs M2, M1 vs M3.
def lr(small, large):
    dchi = small["chi_square"] - large["chi_square"]
    ddf = small["df"] - large["df"]
    if ddf <= 0:
        return None
    return {
        "delta_chi2": round(dchi, 3),
