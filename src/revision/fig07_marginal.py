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
    cov=XtXi@meat@XtXi; se=np.sqrt(np.diag(cov)); return b[1],se[1]
me={}
for col,lab in inds:
    d=fdc.dropna(subset=[col]); b,se=lpm(d[col].values.astype(float),d["stress"].values)
    me[lab]={"marginal_pp":float(b*100),"se_pp":float(se*100),"p":float(2*(1-stats.norm.cdf(abs(b/se))))}
json.dump(me,open(OUT/"fig7b_marginal_effects.json","w"),indent=2)
print("marginal effects (pp per +1SD yield decline):")
for k,v in me.items(): print(f"  {k}: {v['marginal_pp']:+.1f}pp (p={v['p']:.3f})")
# ---- figure ----
fig,(axA,axB)=plt.subplots(1,2,figsize=(13,5))
# Panel A: geographic (lon/lat scatter from gazetteer)
gaz=pd.read_csv("data/raw/census/2023_Gaz_counties_national.txt",sep="\t",dtype=str); gaz.columns=[c.strip() for c in gaz.columns]
gaz["fips"]=gaz["GEOID"].str.zfill(5); gaz["lat"]=pd.to_numeric(gaz["INTPTLAT"],errors="coerce"); gaz["lon"]=pd.to_numeric(gaz["INTPTLONG"],errors="coerce")
m=di[di["farm_dependent"]==1].merge(gaz[["fips","lat","lon"]],on="fips",how="left")
m=m[(m["lon"]>-125)&(m["lon"]<-66)&(m["lat"]>24)&(m["lat"]<50)]
sc=axA.scatter(m["lon"],m["lat"],c=m["n_decline_indicators"],cmap="YlOrRd",s=14,edgecolor="none")
axA.set_title("A  Decline indicators, farming-dependent counties"); axA.set_xticks([]); axA.set_yticks([])
plt.colorbar(sc,ax=axA,shrink=0.7,label="# indicators (of 6)")
# Panel B: marginal effects
labs=list(me.keys()); vals=[me[k]["marginal_pp"] for k in labs]; ses=[1.96*me[k]["se_pp"] for k in labs]
yp=np.arange(len(labs)); cols=["#b2182b" if v>0 else "#2166ac" for v in vals]
axB.barh(yp,vals,xerr=ses,color=cols,alpha=0.85,capsize=3)
axB.set_yticks(yp); axB.set_yticklabels(labs); axB.axvline(0,color="k",lw=0.8)
axB.set_xlabel("Marginal effect on P(indicator), pp per +1 SD yield decline")
axB.set_title("B  Marginal effect of yield decline on each indicator")
fig.tight_layout(); fig.savefig(FIG/"fig07_cascade.pdf",dpi=200); fig.savefig(FIG/"fig07_cascade.png",dpi=150)
print("saved fig07_cascade.pdf (marginal-effects Panel B)")
