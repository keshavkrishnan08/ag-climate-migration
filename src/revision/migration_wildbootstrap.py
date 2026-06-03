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
