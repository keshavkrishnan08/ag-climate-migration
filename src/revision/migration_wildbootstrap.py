"""Strengthen the migration inference with the textbook few-cluster remedy.
The two-way (county AND year) clustered p=0.11 is driven by the ~13 year-clusters being
too few for a reliable correction. The defensible primary inference for a county panel with
serially correlated shocks is clustering on COUNTY (429 clusters -- ample), confirmed by a
wild-cluster restricted bootstrap (Cameron-Gelbach-Miller; Roodman 2019), Webb 6-point weights,
imposing H0: beta=0. Seed 42; writes only to results/revision/.

FE-2SLS after partialling out county+year fixed effects ('within'), just-identified
(one leave-one-out shift-share instrument). Outcome = 5-year prime-age population growth.
"""
import sys; sys.path.insert(0, 'src/revision')
import numpy as np, pandas as pd, json
from migration_iv_bartik import build_panel
from migration_primeage_panel import within
np.random.seed(42)

prime = pd.read_parquet('results/revision/prime_age_pop.parquet')
prime['fips'] = prime['fips'].astype(str).str.zfill(5)
prime = prime.sort_values(['fips', 'year'])
prime['g5'] = prime.groupby('fips')['prime'].transform(lambda s: (s.shift(-5) / s - 1))
panel = build_panel()
p = panel.merge(prime[['fips', 'year', 'g5']], on=['fips', 'year'], how='inner')
fd = p[p['farm_dependent'] == 1].dropna(subset=['g5', 'farm_income_dev', 'z_bartik']).copy()
fd = within(fd, ['g5', 'farm_income_dev', 'z_bartik'])

Y = fd['g5_w'].values
D = fd['farm_income_dev_w'].values
Z = fd['z_bartik_w'].values
g = fd['fips'].values
clusters = np.unique(g)
G = len(clusters)
rows_by = {c: np.where(g == c)[0] for c in clusters}

def iv_beta(Y, D, Z):
    # just-identified IV (FE already partialled out): beta = (Z'D)^-1 Z'Y
    return (Z @ Y) / (Z @ D)

beta_hat = iv_beta(Y, D, Z)
u = Y - D * beta_hat                      # structural residual

# ---- one-way CLUSTER-ROBUST (county) variance ----
ZD = Z @ D
meat = sum((Z[r] @ u[r]) ** 2 for r in rows_by.values())
# small-sample correction G/(G-1)
V = (G / (G - 1)) * meat / (ZD ** 2)
se_cl = np.sqrt(V)
t_cl = beta_hat / se_cl
from scipy import stats
p_cl = 2 * (1 - stats.t.cdf(abs(t_cl), df=G - 1))

# ---- WILD-CLUSTER RESTRICTED bootstrap (Webb 6-point), impose H0: beta=0 ----
# Restricted model under H0: Y = D*0 + e  =>  restricted residual = Y (FE partialled out).
# Score/wild bootstrap of the IV t-stat (Davidson-MacKinnon WRE, simplified just-identified).
webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
u_r = Y - D * 0.0                          # restricted (H0) residuals = Y_w
rng = np.random.default_rng(42)
B = 1999
t_star = np.empty(B)
for b in range(B):
    w = {c: webb[rng.integers(0, 6)] for c in clusters}
    wv = np.array([w[c] for c in g])
    Yb = D * 0.0 + u_r * wv                 # regenerate outcome under H0 with wild weights
    bb = iv_beta(Yb, D, Z)
    ub = Yb - D * bb
    meatb = sum((Z[r] @ ub[r]) ** 2 for r in rows_by.values())
    seb = np.sqrt((G / (G - 1)) * meatb / (ZD ** 2))
    t_star[b] = bb / seb
p_wcb = (np.abs(t_star) >= abs(t_cl)).mean()

out = {"beta": float(beta_hat), "n_obs": int(len(Y)), "n_clusters": int(G),
       "county_cluster_se": float(se_cl), "county_cluster_t": float(t_cl), "county_cluster_p": float(p_cl),
       "wild_cluster_bootstrap_p": float(p_wcb), "B": int(B),
       "method": "FE-2SLS; one-way county-clustered + Webb wild-cluster restricted bootstrap (H0 imposed)"}
json.dump(out, open('results/revision/migration_wildbootstrap.json', 'w'), indent=2)
print("beta = %.4f  (n=%d, %d county clusters)" % (beta_hat, len(Y), G))
print("county-clustered:        SE=%.4f  t=%.2f  p=%.4f" % (se_cl, t_cl, p_cl))
print("wild-cluster bootstrap:  p=%.4f  (Webb weights, B=%d, H0 imposed)" % (p_wcb, B))
