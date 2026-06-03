"""Migration: overidentified multi-instrument 2SLS to tighten precision (#2).

Instead of one combined shift-share shock, use crop-specific leave-one-out
shift-share instruments (corn, soybeans, wheat) as a vector. Overidentification
improves efficiency (tighter CI), and the Hansen J statistic tests instrument
validity. Estimated on high farm-intensity counties. Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from migration_iv_bartik import build_panel, demean2
DATA_PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "revision"
PRICE = {"corn": 5.04, "soybeans": 12.29, "wheat_winter": 6.72, "wheat_spring": 7.38,
         "cotton": 0.93, "sorghum": 4.80, "barley": 5.64, "oats": 3.35}
INSTR_CROPS = ["corn", "soybeans", "wheat_winter", "sorghum"]


def crop_instruments():
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet",
                         columns=["fips", "year", "crop", "yield_anomaly", "acres_harvested"])
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    bmix = fm[fm["year"].between(2000, 2009)].groupby(["fips", "crop"])["acres_harvested"].mean().reset_index()
    tot = bmix.groupby("fips")["acres_harvested"].transform("sum")
    bmix["share"] = bmix["acres_harvested"] / tot.replace(0, np.nan)
    natl = fm.groupby(["crop", "year"])["yield_anomaly"].agg(["sum", "count"]).reset_index()
