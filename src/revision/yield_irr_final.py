"""Final yield model: temperature spectrum + monthly climate + soil + latitude +
IRRIGATION SHARE (sub-county management, from NASS practice records). Trains both
the levels model (AgMIP-comparable, target R^2>=0.5) and the %-deviation model.
Caches the engineered feature matrix so it is built only once. Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_v7_spectrum import temperature_spectrum, es, GROW
from yield_v5_percrop import latitude
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "revision"
SEED = 42
CACHE = OUT / "yield_features_full.parquet"


def build_features():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet",
                         columns=["fips", "year", "crop", "yield_bu_acre",
                                  "log_population", "log_median_income", "yield_trend_slope_15yr"])
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    fm = fm[fm["yield_bu_acre"] > 0].copy()
    m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet")
    m["fips"] = m["fips"].astype(str).str.zfill(5)
    tmax = np.column_stack([(m[f"tmax_m{mm}"] - 32) * 5 / 9 for mm in GROW])
    tmin = np.column_stack([(m[f"tmin_m{mm}"] - 32) * 5 / 9 for mm in GROW])
    spec = temperature_spectrum(tmax, tmin)
    cf = pd.DataFrame({"fips": m["fips"].values, "year": m["year"].values})
    for b in range(spec.shape[1]):
        cf[f"tbin_{b}"] = spec[:, b]
    for mm in GROW:
        cf[f"precip_{mm}"] = m[f"precip_m{mm}"].values
        cf[f"pdsi_{mm}"] = m[f"pdsi_m{mm}"].values
        tc = (m[f"tmax_m{mm}"] - 32) * 5 / 9; tn = (m[f"tmin_m{mm}"] - 32) * 5 / 9
        cf[f"vpd_{mm}"] = (es(tc) - es(tn)).clip(lower=0).values
    panel = fm.merge(cf, on=["fips", "year"], how="left").merge(latitude(), on="fips", how="left")
    cmax = panel.groupby(["fips", "crop"])["yield_bu_acre"].transform("max")
    natmax = panel.groupby("crop")["yield_bu_acre"].transform("max")
    panel["nccpi"] = (cmax / natmax).clip(0, 1)
    # irrigation propensity (time-invariant, projectable); 0 = rainfed
    irr = pd.read_parquet(OUT / "irrigation_share.parquet")[["fips", "crop", "irr_prop"]].drop_duplicates()
    panel = panel.merge(irr, on=["fips", "crop"], how="left")
    panel["irr_prop"] = panel["irr_prop"].fillna(0.0)
    panel.to_parquet(CACHE, index=False)
    return panel


def trend_pct(panel):
    tr = panel[panel["year"] <= 2012]
    g = tr.groupby(["fips", "crop"])
    agg = g.agg(n=("year", "size"), sx=("year", "sum"), sy=("yield_bu_acre", "sum"),
                sxx=("year", lambda s: (s.astype(float)**2).sum())).reset_index()
    sxy = (tr.assign(xy=tr["year"]*tr["yield_bu_acre"]).groupby(["fips", "crop"])["xy"]
