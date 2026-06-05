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
    if tr.sum()<500 or te.sum()<100: continue
    X=d[feats].fillna(0); y=d["yield_bu_acre"]
    gb=lgb.LGBMRegressor(objective="regression",n_estimators=2500,learning_rate=0.02,max_depth=8,
        num_leaves=127,min_child_samples=30,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.05,reg_lambda=0.5,
        random_state=42,verbose=-1).fit(X[tr],y[tr])
    sc=StandardScaler().fit(X[tr]); mlp=MLPRegressor(hidden_layer_sizes=(256,128,64),activation="relu",
        alpha=1e-3,max_iter=300,random_state=42,early_stopping=True).fit(sc.transform(X[tr]),y[tr])
    Pb=np.column_stack([gb.predict(X[bl]),mlp.predict(sc.transform(X[bl]))])
    w,_=nnls(Pb,y[bl].values); w=w/w.sum() if w.sum()>0 else np.array([1,0])
    pr=w[0]*gb.predict(X[te])+w[1]*mlp.predict(sc.transform(X[te])); yt=y[te].values
