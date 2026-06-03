"""Clean opportunity recompute with a real cropland denominator (removes the
all-farm-acreage inflation, e.g. orchards in Yakima).

Expandable acreage is capped at the county's historical MAXIMUM total harvested
acreage (land that has actually been cropped) plus a 15% idle-cropland margin,
instead of all-farm 'ACRES OPERATED' x a state fraction. Opportunity is reported
as NET farm income (22% USDA-ERS grain/oilseed operating margin). Seed 42.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "revision"
PRICE = {"corn": 5.04, "soybeans": 12.29, "wheat_winter": 6.72, "wheat_spring": 7.38,
         "cotton": 0.93, "sorghum": 4.80, "barley": 5.64, "oats": 3.35}
MIN_VIABLE = {"corn": 60, "soybeans": 20, "wheat_winter": 30, "wheat_spring": 20,
              "barley": 30, "oats": 25, "sorghum": 40}
MARGIN = 0.22
EXPANSION_HEADROOM = 1.15   # allow 15% onto historically idle cropland


def main():
    front = pd.read_csv(ROOT / "results" / "frontier" / "opportunity_counties_SSP245.csv",
                        dtype={"fips": str})
    front["fips"] = front["fips"].str.zfill(5)
    fm = pd.read_parquet(ROOT / "data" / "processed" / "feature_matrix.parquet",
                         columns=["fips", "year", "crop", "acres_harvested"])
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    # historical max total harvested acreage per county (real cropland ceiling)
    yr_tot = fm.groupby(["fips", "year"])["acres_harvested"].sum()
    cropland_ceiling = yr_tot.groupby("fips").max().rename("cropland_ceiling")
    cur_harv = (fm[fm["year"].between(2019, 2023)].groupby(["fips", "year"])["acres_harvested"]
                .sum().groupby("fips").mean().rename("cur_harv"))
    cap = pd.concat([cropland_ceiling, cur_harv], axis=1).reset_index()
    cap["expandable_clean"] = (cap["cropland_ceiling"] * EXPANSION_HEADROOM
                               - cap["cur_harv"]).clip(lower=0)

    yp = pd.read_parquet(ROOT / "data" / "projections" / "yield_projections_SSP245.parquet",
                         columns=["fips", "year", "crop", "yield_projected"])
    yp["fips"] = yp["fips"].astype(str).str.zfill(5)
    late = yp[yp["year"].between(2038, 2042)].groupby(["fips", "crop"])["yield_projected"].mean().reset_index()
    late["price"] = late["crop"].map(PRICE).fillna(4.0)
    late["min_v"] = late["crop"].map(MIN_VIABLE).fillna(20)
