"""Stacked levels ensemble: gradient-boosted trees + MLP (NNLS blend), per crop,
on cached features incl. irrigation + lagged yield. The hybrid-network architecture
the benchmark papers use. Reports levels R2 per crop. Seed 42."""
import json, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from scipy import stats
from scipy.optimize import nnls
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent.parent.parent; OUT=ROOT/"results"/"revision"
p=pd.read_parquet(OUT/"yield_features_full.parquet"); p["fips"]=p["fips"].astype(str).str.zfill(5)
p=p.sort_values(["fips","crop","year"])
for k in (1,2,3): p[f"lag{k}"]=p.groupby(["fips","crop"])["yield_bu_acre"].shift(k)
p["lag_mean3"]=p[["lag1","lag2","lag3"]].mean(axis=1); p["tt"]=p["year"]-1980
clim=[c for c in p.columns if c.startswith(("tbin_","precip_","pdsi_","vpd_"))]
feats=clim+["latitude","nccpi","irr_prop","log_population","log_median_income","yield_trend_slope_15yr","lag_mean3","tt"]
res={}; ao=[]; ap=[]
for crop in sorted(p["crop"].unique()):
    d=p[(p["crop"]==crop)&p["lag_mean3"].notna()]
    tr=d["year"]<=2009; bl=(d["year"]>2009)&(d["year"]<=2012); te=(d["year"]>2012)&(d["year"]<=2023)
