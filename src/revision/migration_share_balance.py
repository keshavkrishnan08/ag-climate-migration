"""Shift-share identification check (Goldsmith-Pinkham 2020; Borusyak-Hull-Jaravel 2022).
The leave-one-out Bartik instrument is z_it = sum_c share_ic(2000-09 baseline) * price_c *
g_loo_{c,t}. Identification rests either on baseline-share exogeneity or on shock exogeneity
(the shocks are OTHER counties' national crop-yield innovations, plausibly exogenous to a single
county). We test the share-exogeneity leg directly: if baseline shares were correlated with a
county's pre-existing demographic trajectory, the instrument would proxy a pre-trend. So we
regress PRE-PERIOD (pre-estimation-window) population growth on the county's instrument exposure.
A null = no pre-trend = the instrument is not aligned with pre-existing differential trends.
Seed 42; writes only to results/revision/.
"""
import sys; sys.path.insert(0, 'src/revision')
import json
import numpy as np, pandas as pd
from scipy import stats
from migration_iv_bartik import build_panel
np.random.seed(42)
OUT = __import__('pathlib').Path('results/revision')

def hc1(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; XtXi = np.linalg.inv(X.T @ X)
    cov = XtXi @ ((X * (r ** 2)[:, None]).T @ X) @ XtXi * (len(y) / (len(y) - X.shape[1]))
    return b, np.sqrt(np.diag(cov))

panel = build_panel()
panel['fips'] = panel['fips'].astype(str).str.zfill(5)

# county instrument exposure = mean over the 2009-2023 estimation window
instr = panel[panel.year.between(2009, 2023)].groupby('fips')['z_bartik'].mean().rename('instr')

# PRE-PERIOD population growth (1990 -> 2000), strictly before the 2000-09 baseline-share window
pop = panel[['fips', 'year', 'total_population']].dropna()
def pop_yr(y):
    return pop[pop.year == y].set_index('fips')['total_population']
pre = (pop_yr(2000) / pop_yr(1990) - 1).rename('pretrend_9000')
# also 2000->2008 as a second pre-window (overlaps baseline shares but precedes outcome window)
pre2 = (pop_yr(2008) / pop_yr(2000) - 1).rename('pretrend_0008')

fd = panel[panel.farm_dependent == 1]['fips'].unique()
d = pd.concat([instr, pre, pre2], axis=1).dropna(subset=['instr'])
d = d[d.index.isin(set(fd))]

def reg(col):
    dd = d[['instr', col]].replace([np.inf, -np.inf], np.nan).dropna()
    z = (dd['instr'] - dd['instr'].mean()) / dd['instr'].std()
    X = np.column_stack([np.ones(len(dd)), z.values])
    b, se = hc1(dd[col].astype(float).values, X)
    t = b[1] / se[1]; p = 2 * (1 - stats.norm.cdf(abs(t)))
    return {"beta_per_1sd_instr": float(b[1]), "se": float(se[1]), "p": float(p), "n": int(len(dd))}

r9000 = reg('pretrend_9000')
r0008 = reg('pretrend_0008')
out = {"test": "pre-trend balance: pre-period population growth ~ instrument exposure (farming-dependent)",
       "interpretation": "null coefficient => instrument not aligned with pre-existing trends => share-exogeneity supported",
       "pretrend_1990_2000": r9000, "pretrend_2000_2008": r0008,
       "refs": "Goldsmith-Pinkham et al. 2020; Borusyak, Hull & Jaravel 2022"}
json.dump(out, open(OUT / "migration_share_balance.json", "w"), indent=2)
print("Shift-share pre-trend balance test (farming-dependent counties):")
print("  pre-period 1990-2000 pop growth ~ instrument:  beta/1sd=%+.4f  p=%.3f  n=%d"
      % (r9000["beta_per_1sd_instr"], r9000["p"], r9000["n"]))
print("  pre-period 2000-2008 pop growth ~ instrument:  beta/1sd=%+.4f  p=%.3f  n=%d"
      % (r0008["beta_per_1sd_instr"], r0008["p"], r0008["n"]))
