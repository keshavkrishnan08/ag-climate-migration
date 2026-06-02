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
def hc1(y,X):
    b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b; XtXi=np.linalg.inv(X.T@X)
    cov=XtXi@((X*(r**2)[:,None]).T@X)@XtXi*(len(y)/(len(y)-X.shape[1])); se=np.sqrt(np.diag(cov))
    return b,se

# county net insurance flow (2040-2050): + = underpriced/recipient, - = overpriced/payer
cy=pd.read_parquet(OUT/"insurance_mispricing_county_year.parquet")
cyn=cy[cy.year.between(2040,2050)].groupby("fips")["flow_tay"].mean().rename("net_flow").reset_index()
cyn["fips"]=cyn["fips"].astype(str).str.zfill(5)
front=pd.read_csv("results/frontier/opportunity_counties_SSP245.csv",dtype={"fips":str}); front["fips"]=front["fips"].str.zfill(5)
