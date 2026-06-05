"""Revision v4: push anomaly R^2 with additional process-relevant features.

Builds on v3 (VPD, EDD30, heat-days, soil-moisture stress) by adding the thermal
and critical-period precipitation features standard in the modern crop-yield
literature but absent from the original 50-feature set:
  kdd34_growing   : killing degree-days above 34 C (lethal heat, distinct from EDD>30)
  precip_jul_anom : July precipitation anomaly (grain-fill / silking water stress)
  precip_aug_anom : August precipitation anomaly
  dtr_growing_anom: diurnal temperature range anomaly (cloud/heat-stress signal)
  vpd_aug         : August vapour pressure deficit
Trains LightGBM only (the v3 ensemble blend collapsed to LightGBM, w=[1,0,0]),
so this is the operative model. Same split (train<=2012, test 2013-2023). Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_model_v3_features import build_modern_features, add_county_anomalies, es_kpa, GROW_MONTHS

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "revision"
SEED = 42


def extra_features():
    m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet")
    m["fips"] = m["fips"].astype(str).str.zfill(5)
    for mm in GROW_MONTHS:
        m[f"tmaxc_{mm}"] = (m[f"tmax_m{mm}"] - 32) * 5 / 9
        m[f"tminc_{mm}"] = (m[f"tmin_m{mm}"] - 32) * 5 / 9
    # KDD>34C (sinusoid quadrature)
    thr = 34.0; phase = np.linspace(0, np.pi, 24); kdd = np.zeros(len(m)); dtr = np.zeros(len(m))
    for mm in GROW_MONTHS:
        tmn = m[f"tminc_{mm}"].values; tmx = m[f"tmaxc_{mm}"].values
        mid = (tmx + tmn) / 2; amp = (tmx - tmn) / 2
        dd = np.zeros_like(mid)
