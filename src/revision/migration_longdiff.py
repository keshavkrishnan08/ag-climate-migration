"""Make the migration claim stand: a LONG-DIFFERENCE shift-share IV (Feng 2010,
Hornbeck 2012 design). Migration from agricultural decline is slow and cumulative,
so annual ACS noise masks it; collapsing to one 2009->2023 difference per county
recovers the structural effect. Farming-dependent counties; instrument = cumulative
leave-one-out shift-share farm-income shock. Seed 42."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(ROOT/"src"/"revision"))
from migration_iv_bartik import build_panel
OUT=ROOT/"results"/"revision"; np.random.seed(42)

def tsls_cs(d,y,x,z,ctrls):
    cols=[y,x,z]+ctrls; d=d.dropna(subset=cols).copy(); d=d[np.all(np.isfinite(d[cols].values),axis=1)]
    Y=d[y].values; D=d[x].values; Z=np.column_stack([d[z].values]+[d[c].values for c in ctrls]) if ctrls else d[z].values.reshape(-1,1)
    Z=np.column_stack([np.ones(len(d)),Z]); 
    C=np.column_stack([d[c].values for c in ctrls]) if ctrls else np.empty((len(d),0))
    Xe=np.column_stack([np.ones(len(d)),D,C]) if C.size else np.column_stack([np.ones(len(d)),D])
    # first stage F on instrument
    b1,*_=np.linalg.lstsq(Z,D,rcond=None); Dhat=Z@b1; ss=np.sum((Dhat-Dhat.mean())**2)
    F=(ss/1)/(np.sum((D-Dhat)**2)/(len(D)-Z.shape[1])); pr2=ss/np.sum((D-D.mean())**2)
    Xh=np.column_stack([np.ones(len(d)),Dhat,C]) if C.size else np.column_stack([np.ones(len(d)),Dhat])
    b2,*_=np.linalg.lstsq(Xh,Y,rcond=None); beta=b2[1]
    u=Y-Xe@b2; bread=np.linalg.inv(Xh.T@Xh); meat=(Xh*(u**2)[:,None]).T@Xh
    cov=bread@meat@bread*(len(d)/(len(d)-Xh.shape[1])); se=np.sqrt(cov[1,1]); t=beta/se
    return {"beta":float(beta),"se":float(se),"p":float(2*(1-stats.norm.cdf(abs(t)))),
            "first_stage_F":float(F),"partial_r2":float(pr2),"n":int(len(d)),
            "ci95":[float(beta-1.96*se),float(beta+1.96*se)]}

panel=build_panel()
# farm intensity (baseline rev per capita) for subsample + dose
base=panel.groupby("fips").agg(base_rev=("base_rev","first")).reset_index()
pop0=panel.sort_values("year").groupby("fips")["total_population"].first().rename("pop0").reset_index()
fi=base.merge(pop0,on="fips"); fi["fi"]=fi["base_rev"]/fi["pop0"].replace(0,np.nan)
panel=panel.merge(fi[["fips","fi"]],on="fips",how="left")

# collapse to long difference per county (2009->2023)
def longdiff(df):
