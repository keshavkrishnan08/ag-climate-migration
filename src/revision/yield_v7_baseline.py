"""Fair comparison: same %-deviation target, OLD growing-season aggregate features
(no temperature spectrum). Isolates how much the spectrum representation adds vs
the original aggregates, holding the target fixed.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_v5_percrop import latitude
DATA_PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "revision"
SEED = 42


def main():
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet")
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    fm = fm[fm["yield_bu_acre"] > 0].copy()
    # same %-deviation target as v7 (closed-form trend on <=2012)
    tr = fm[fm["year"] <= 2012]
    g = tr.groupby(["fips", "crop"])
    agg = g.agg(n=("year", "size"), sx=("year", "sum"), sy=("yield_bu_acre", "sum"),
                sxx=("year", lambda s: (s.astype(float)**2).sum())).reset_index()
    sxy = (tr.assign(xy=tr["year"]*tr["yield_bu_acre"]).groupby(["fips", "crop"])["xy"]
           .sum().reset_index(name="sxy"))
    agg = agg.merge(sxy, on=["fips", "crop"])
    den = agg["n"]*agg["sxx"] - agg["sx"]**2
    agg["slope"] = np.where(den != 0, (agg["n"]*agg["sxy"] - agg["sx"]*agg["sy"])/den, np.nan)
    agg["intercept"] = (agg["sy"] - agg["slope"]*agg["sx"])/agg["n"]
    agg = agg[(agg["n"] >= 8) & agg["slope"].notna()]
    fm = fm.merge(agg[["fips", "crop", "slope", "intercept"]], on=["fips", "crop"], how="inner")
    fm["trend"] = fm["intercept"] + fm["slope"]*fm["year"]
    fm = fm[fm["trend"] > 0].copy()
    fm["dev_pct"] = (fm["yield_bu_acre"]/fm["trend"] - 1).clip(-1, 1)
    fm = fm.merge(latitude(), on="fips", how="left")
    cmax = fm.groupby(["fips", "crop"])["yield_bu_acre"].transform("max")
    natmax = fm.groupby("crop")["yield_bu_acre"].transform("max")
