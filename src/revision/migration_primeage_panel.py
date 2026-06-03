"""Prime-age (25-54) population in the WITHIN-COUNTY panel-FE shift-share IV — the
design that holds for total population (the long-difference fails only because it is
a cross-section with no fixed effects). Outcome: 3-yr-forward prime-age growth.
County+year FE; instrument = leave-one-out shift-share. High-farm-intensity tercile
of farming-dependent counties + placebo. Seed 42."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(ROOT/"src"/"revision"))
from migration_iv_bartik import build_panel
OUT=ROOT/"results"/"revision"; np.random.seed(42)

def within(df,cols,gi="fips",gt="year"):
    o=df.copy()
    for c in cols:
        s=o[c].astype(float)
        for _ in range(30): s=s-s.groupby(o[gi]).transform("mean"); s=s-s.groupby(o[gt]).transform("mean")
        o[c+"_w"]=s
    return o
def fe2sls(d,y,x,z,cl="fips"):
    d=d.dropna(subset=[y,x,z]).copy(); d=d[np.all(np.isfinite(d[[y,x,z]].values),axis=1)]
    d=within(d,[y,x,z])
    Z=d[z+"_w"].values.reshape(-1,1); D=d[x+"_w"].values; Y=d[y+"_w"].values
    Zc=np.column_stack([np.ones(len(d)),Z])
    b1,*_=np.linalg.lstsq(Zc,D,rcond=None); Dhat=Zc@b1
    F=(np.sum((Dhat-Dhat.mean())**2))/ (np.sum((D-Dhat)**2)/(len(D)-2))
    Xh=np.column_stack([np.ones(len(d)),Dhat]); b2,*_=np.linalg.lstsq(Xh,Y,rcond=None); beta=b2[1]
    u=Y-np.column_stack([np.ones(len(d)),D])@b2
    # cluster-robust by county
    bread=np.linalg.inv(Xh.T@Xh); meat=np.zeros((2,2))
    for _,idx in d.groupby(cl).indices.items():
        Xg=Xh[idx]; ug=u[idx]; meat+=Xg.T@np.outer(ug,ug)@Xg
    cov=bread@meat@bread; se=np.sqrt(cov[1,1]); t=beta/se
    return {"beta":float(beta),"se":float(se),"p":float(2*(1-stats.norm.cdf(abs(t)))),
            "first_stage_F":float(F),"n":int(len(d)),"n_cty":int(d[cl].nunique()),
            "ci95":[float(beta-1.96*se),float(beta+1.96*se)]}

prime=pd.read_parquet(OUT/"prime_age_pop.parquet"); prime["fips"]=prime["fips"].astype(str).str.zfill(5)
prime=prime.sort_values(["fips","year"])
prime["prime_growth_3yr"]=prime.groupby("fips")["prime"].transform(lambda s:(s.shift(-3)/s-1))
panel=build_panel()
base=panel.groupby("fips").agg(base_rev=("base_rev","first")).reset_index()
pop0=panel.sort_values("year").groupby("fips")["total_population"].first().rename("pop0").reset_index()
fi=base.merge(pop0,on="fips"); fi["fi"]=fi["base_rev"]/fi["pop0"].replace(0,np.nan)
p=panel.merge(prime[["fips","year","prime_growth_3yr"]],on=["fips","year"],how="inner").merge(fi[["fips","fi"]],on="fips",how="left")
fd=p[p["farm_dependent"]==1]
out={}
out["primeage_panelFE_farmdep"]=fe2sls(fd,"prime_growth_3yr","farm_income_dev","z_bartik")
thr=fd["fi"].quantile(0.667)
out["primeage_panelFE_high_intensity"]=fe2sls(fd[fd["fi"]>=thr],"prime_growth_3yr","farm_income_dev","z_bartik")
out["primeage_panelFE_placebo_low"]=fe2sls(fd[fd["fi"]<fd["fi"].quantile(0.333)],"prime_growth_3yr","farm_income_dev","z_bartik")
b=out["primeage_panelFE_high_intensity"]["beta"]
out["interp"]=f"sustained 10% farm-income decline -> {b*-0.10*100:+.2f}% 3yr prime-age growth (high-intensity)"
json.dump(out,open(OUT/"migration_primeage_panel.json","w"),indent=2)
for k,v in out.items():
    if isinstance(v,dict): print(f"  {k}: beta={v['beta']:+.3f} p={v['p']:.4f} F={v['first_stage_F']:.0f} n={v['n']} ({v['n_cty']} cty)")
print(" ",out["interp"])
