"""Close the magnitude question: test the migration claim on PRIME-AGE (25-54)
population, the outcome that responds to economic shocks (total population is diluted
by retirees, children, births/deaths). Prime-age county panel from Census PEP
(2010-2019 alldata + 2020-2023 agesex), no API key. Pre-COVID long difference
2010->2019 with the leave-one-out shift-share IV; farming-dependent counties. Seed 42."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(ROOT/"src"/"revision"))
from migration_iv_bartik import build_panel
from migration_longdiff import tsls_cs
OUT=ROOT/"results"/"revision"; np.random.seed(42)

prime=pd.read_parquet(OUT/"prime_age_pop.parquet"); prime["fips"]=prime["fips"].astype(str).str.zfill(5)
panel=build_panel()
base=panel.groupby("fips").agg(base_rev=("base_rev","first")).reset_index()
pop0=panel.sort_values("year").groupby("fips")["total_population"].first().rename("pop0").reset_index()
fi=base.merge(pop0,on="fips"); fi["fi"]=fi["base_rev"]/fi["pop0"].replace(0,np.nan)
# cumulative instrument & treatment over 2010-2019 (pre-COVID)
pp=panel[panel["year"].between(2010,2019)].groupby("fips").agg(
    cum_fid=("farm_income_dev","mean"), cum_z=("z_bartik","mean"),
    winter=("winter_tmin_anom","mean"), fdep=("farm_dependent","max")).reset_index()
pp=pp.merge(fi[["fips","fi"]],on="fips",how="left")
# prime-age long difference 2010->2019
p10=prime[prime.year==2010][["fips","prime"]].rename(columns={"prime":"p10"})
p19=prime[prime.year==2019][["fips","prime"]].rename(columns={"prime":"p19"})
pa=p10.merge(p19,on="fips"); pa=pa[(pa.p10>0)&(pa.p19>0)]
pa["dlog_prime"]=np.log(pa["p19"])-np.log(pa["p10"])
cs=pp.merge(pa[["fips","dlog_prime"]],on="fips",how="inner").dropna(subset=["dlog_prime","cum_fid","cum_z"])
out={}
out["primeage_longdiff_farmdep"]=tsls_cs(cs[cs.fdep==1],"dlog_prime","cum_fid","cum_z",["winter"])
hi=cs[(cs.fdep==1)&(cs.fi>=cs[cs.fdep==1].fi.quantile(0.5))]
out["primeage_longdiff_high_intensity"]=tsls_cs(hi,"dlog_prime","cum_fid","cum_z",["winter"])
out["primeage_placebo_nonfarm"]=tsls_cs(cs[cs.fdep==0],"dlog_prime","cum_fid","cum_z",["winter"])
# economic magnitude: effect of a sustained 10% farm-income decline on prime-age pop
b=out["primeage_longdiff_farmdep"]["beta"]
out["interpretation"]=f"A sustained 10%% farm-income decline -> {b*-0.10*100:+.2f}%% prime-age population over 2010-2019"
json.dump(out,open(OUT/"migration_primeage.json","w"),indent=2)
for k,v in out.items():
    if isinstance(v,dict): print(f"  {k}: beta={v['beta']:+.3f} p={v['p']:.4f} F={v['first_stage_F']:.1f} CI={[round(x,3) for x in v['ci95']]} n={v['n']}")
print(" ",out["interpretation"])
