"""R1-2f: the direct chain crop yields -> farm income -> local fiscal capacity,
in farming-dependent counties. Reviewer 1 suggested focusing on this more-defensible
relationship. We estimate, with county + year fixed effects (cluster-robust SE):
  Link 1: farm revenue on lagged county yield (acreage-weighted) -- the income channel.
  Link 2a: farmland value (the agricultural property-tax base) on farm revenue.
  Link 2b: median household income on farm revenue.
The farmland-value link is the fiscal mechanism: ag property tax = assessed land
value x rate, so a farm-income-driven decline in land value contracts the local tax
base (Census 2025). Seed 42."""
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
PRICE={"corn":5.04,"soybeans":12.29,"wheat_winter":6.72,"wheat_spring":7.38,"cotton":0.93,"sorghum":4.80,"barley":5.64,"oats":3.35}
