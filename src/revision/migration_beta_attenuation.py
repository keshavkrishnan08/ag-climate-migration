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
import numpy as np

OUT = Path("results/revision")
rng = np.random.default_rng(42)

# Headline MC inputs (mirroring migration_depop_montecarlo.json).
n_draws = 200_000
horizon = np.arange(2025, 2051)  # 26 years
prime_age_base = 1_130_330
beta_hat = 0.0491
beta_se = 0.0149
income_decline_lo, income_decline_hi = 0.15, 0.25
household_lo, household_hi = 2.2, 2.6
per_capita_lo, per_capita_hi = 70_000.0, 75_000.0
multiplier_lo, multiplier_hi = 1.6, 1.8
discount_lo, discount_hi = 0.03, 0.05


def attenuation_path(start, end, t_target=2040, t_end=2050):
    out = []
    for year in horizon:
        if year <= t_target:
            frac = (year - 2024) / (t_target - 2024)
        else:
            frac = 1.0
        out.append(start + frac * (end - start))
    return np.array(out)


def mc_npv(beta_path_fn):
    # Antithetic variates: pair (u, 1-u).
    half = n_draws // 2
    z = rng.standard_normal(half)
    beta0 = np.clip(beta_hat + beta_se * z, 0.0, None)
    beta0 = np.concatenate([beta0, np.clip(beta_hat - beta_se * z, 0.0, None)])

    u_inc = rng.uniform(income_decline_lo, income_decline_hi, n_draws)
    u_hh = rng.uniform(household_lo, household_hi, n_draws)
    u_pc = rng.uniform(per_capita_lo, per_capita_hi, n_draws)
    u_m = rng.uniform(multiplier_lo, multiplier_hi, n_draws)
    u_d = rng.uniform(discount_lo, discount_hi, n_draws)

    # Linearly accumulate displaced prime-age over the horizon, then value.
    yrs = horizon - 2024
    Hsq = len(yrs)
    npvs = np.zeros(n_draws)
    for i in range(0, n_draws, 5000):
        sl = slice(i, i + 5000)
        b = beta0[sl][:, None]              # (chunk, 1)
        inc = u_inc[sl][:, None]
        hh = u_hh[sl][:, None]
        pc = u_pc[sl][:, None]
        m = u_m[sl][:, None]
        d = u_d[sl][:, None]

        # Time-varying beta multiplier: beta(t) / beta_hat.
        bt = beta_path_fn() / beta_hat        # (years,)
        D_y = b * bt[None, :] * inc * prime_age_base * (yrs[None, :] / Hsq)
        flow = D_y * hh * pc * m
        disc = (1 + d) ** yrs[None, :]
