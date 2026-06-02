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
