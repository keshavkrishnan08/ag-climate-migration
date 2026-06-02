"""Make the 'unified, mutually-reinforcing framework' EMPIRICAL: estimate the three
inter-finding links the Discussion asserts.
  L1 Insurance -> frontier: share of the overpriced (net-payer) insurance outflow that
     originates in the 514 northern frontier counties (capital drained from where it is needed).
  L2 Stranded -> decline: does capitalized climate exposure predict the count of rural-decline
     indicators among farming-dependent counties? (cross-section, HC1 SE)
  L3 Insurance -> adaptation: do insurance-underpriced (subsidized, climate-stressed) counties
     adapt LESS (lower crop-switching) conditional on climate stress? (HC1 SE)
Seed 42; writes only to results/revision/.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT=Path("."); OUT=ROOT/"results/revision"
def fdep():
    cc=pd.read_csv("data/raw/other/ers_atlas/CountyClassifications.csv",dtype=str,encoding="latin-1")
    cc=cc.rename(columns={cc.columns[0]:"fips"}); cc["fips"]=cc["fips"].str.zfill(5)
    f=cc[cc["Attribute"]=="Type_2015_Farming_NO"][["fips","Value"]]; f["fdep"]=(f["Value"]=="1").astype(int)
    return f[["fips","fdep"]]
