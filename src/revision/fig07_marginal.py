"""R1-v: regenerate Fig 7B as a MARGINAL-EFFECTS panel (not coincident counts).
Panel A: geographic distribution of decline-indicator count in farming-dependent
counties. Panel B: marginal effect of a 1-SD adverse yield trend on each decline
indicator (linear probability model, farming-dependent counties, HC1 SE)."""
import numpy as np, pandas as pd, json
from pathlib import Path
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("."); OUT=ROOT/"results/revision"; FIG=ROOT/"results/figures_revision"
di=pd.read_csv("data/published_dataset/county_decline_indicators.csv",dtype={"fips":str})
di["fips"]=di["fips"].str.zfill(5)
fd=pd.read_csv("data/raw/other/ers_atlas/CountyClassifications.csv",dtype=str,encoding="latin-1").rename(columns=lambda c:c)
fd=fd.rename(columns={fd.columns[0]:"fips"}); fd["fips"]=fd["fips"].str.zfill(5)
fdf=fd[fd["Attribute"]=="Type_2015_Farming_NO"][["fips","Value"]]; fdf["farm_dependent"]=(fdf["Value"]=="1").astype(int)
di=di.merge(fdf[["fips","farm_dependent"]],on="fips",how="left"); di["farm_dependent"]=di["farm_dependent"].fillna(0).astype(int)
# county yield-stress = -(acreage-weighted 15yr yield trend slope), standardized
fm=pd.read_parquet("data/processed/feature_matrix.parquet",columns=["fips","crop","year","yield_trend_slope_15yr","acres_harvested"])
fm["fips"]=fm["fips"].astype(str).str.zfill(5)
recent=fm[fm["year"]>=2018]
g=recent.groupby("fips").apply(lambda d: np.average(d["yield_trend_slope_15yr"].fillna(0),weights=d["acres_harvested"].clip(lower=1e-6)) if d["acres_harvested"].sum()>0 else d["yield_trend_slope_15yr"].mean(),include_groups=False).rename("yld_trend").reset_index()
di=di.merge(g,on="fips",how="left")
fdc=di[(di["farm_dependent"]==1)&di["yld_trend"].notna()].copy()
fdc["stress"]=-(fdc["yld_trend"]-fdc["yld_trend"].mean())/fdc["yld_trend"].std()  # +1 = declining yields
inds=[("pop_decline","Population decline"),("income_decline","Income decline"),("outmigration","Out-migration"),("school_decline","School decline"),("hospital_closure","Hospital closure")]
def lpm(y,x):
    X=np.column_stack([np.ones(len(x)),x]); b,*_=np.linalg.lstsq(X,y,rcond=None)
    resid=y-X@b; XtXi=np.linalg.inv(X.T@X); meat=(X*(resid**2)[:,None]).T@X
