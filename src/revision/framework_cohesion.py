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
fset=set(front["fips"])
# L1: overpriced outflow total vs from frontier counties
over=cyn[cyn["net_flow"]<0].copy(); over["outflow"]=-over["net_flow"]
total_outflow=over["outflow"].sum(); frontier_outflow=over[over["fips"].isin(fset)]["outflow"].sum()
L1={"total_overpriced_outflow_B":float(total_outflow/1e9),
    "from_frontier_counties_B":float(frontier_outflow/1e9),
    "frontier_share_pct":float(100*frontier_outflow/total_outflow) if total_outflow>0 else None,
    "n_frontier_overpriced":int(over["fips"].isin(fset).sum())}

# L2: stranded exposure -> decline indicators (farming-dependent)
st=pd.read_parquet(OUT/"stranded_central_floored.parquet")[["fips","stranded_value_total","total_acres"]]
st["fips"]=st["fips"].astype(str).str.zfill(5)
st["stranded_per_acre"]=st["stranded_value_total"]/st["total_acres"].replace(0,np.nan)
di=pd.read_csv("data/published_dataset/county_decline_indicators.csv",dtype={"fips":str}); di["fips"]=di["fips"].str.zfill(5)
d2=di.merge(st,on="fips",how="inner").merge(fdep(),on="fips",how="left"); d2["fdep"]=d2["fdep"].fillna(0).astype(int)
d2=d2[(d2["fdep"]==1)&d2["stranded_per_acre"].notna()&np.isfinite(d2["stranded_per_acre"])]
d2["sp_std"]=(d2["stranded_per_acre"]-d2["stranded_per_acre"].mean())/d2["stranded_per_acre"].std()
b,se=hc1(d2["n_decline_indicators"].astype(float).values,np.column_stack([np.ones(len(d2)),d2["sp_std"].values]))
L2={"beta_per_1sd_stranded":float(b[1]),"se":float(se[1]),"p":float(2*(1-stats.norm.cdf(abs(b[1]/se[1])))),"n":int(len(d2))}

# L3: insurance underpricing -> less crop switching, controlling for climate stress
fm=pd.read_parquet("data/processed/feature_matrix.parquet",columns=["fips","year","switching_rate_5yr","yield_anomaly"])
fm["fips"]=fm["fips"].astype(str).str.zfill(5)
sw=fm[fm.year>=2015].groupby("fips").agg(switch=("switching_rate_5yr","mean")).reset_index()
# climate stress = projected decline per county (from stranded per acre, proxy) ; use stranded_per_acre as stress
m=sw.merge(cyn,on="fips",how="inner").merge(st[["fips","stranded_per_acre"]],on="fips",how="left").merge(fdep(),on="fips",how="left")
m["fdep"]=m["fdep"].fillna(0).astype(int)
m=m[(m["fdep"]==1)&m["switch"].notna()&m["stranded_per_acre"].notna()&np.isfinite(m["stranded_per_acre"])]
m["under"]=(m["net_flow"]>0).astype(float)   # 1 = underpriced/subsidized
m["stress_std"]=(m["stranded_per_acre"]-m["stranded_per_acre"].mean())/m["stranded_per_acre"].std()
X=np.column_stack([np.ones(len(m)),m["under"].values,m["stress_std"].values])
b,se=hc1(m["switch"].astype(float).values,X)
L3={"beta_underpriced_on_switching":float(b[1]),"se":float(se[1]),"p":float(2*(1-stats.norm.cdf(abs(b[1]/se[1])))),
    "controls":"climate stress (stranded per acre)","n":int(len(m)),
    "interpretation":"negative beta => subsidized/underpriced counties switch crops less (insurance impedes adaptation)"}
out={"L1_insurance_to_frontier":L1,"L2_stranded_to_decline":L2,"L3_insurance_to_adaptation":L3}
json.dump(out,open(OUT/"framework_cohesion.json","w"),indent=2)
print("L1 insurance->frontier: %.1f%% of overpriced outflow ($%.2fB) comes from the 514 frontier counties (n=%d)"%(L1["frontier_share_pct"],L1["from_frontier_counties_B"],L1["n_frontier_overpriced"]))
print("L2 stranded->decline: +%.2f more decline indicators per 1 SD stranded exposure (p=%.4f, n=%d)"%(L2["beta_per_1sd_stranded"],L2["p"],L2["n"]))
print("L3 insurance->adaptation: underpriced counties switch %.4f %s (p=%.4f, n=%d)"%(L3["beta_underpriced_on_switching"],"LESS" if L3["beta_underpriced_on_switching"]<0 else "MORE",L3["p"],L3["n"]))
