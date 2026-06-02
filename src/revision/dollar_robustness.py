"""Dollar-robustness to yield-model specification (Reviewer 2's explicit alternative:
"convince me that a more sophisticated specification leaves the dollar conclusions
unchanged").

We compute county-level conservative stranded value under two very different yield
specifications -- the statistical ML model and a purely process-based Schlenker-
Roberts extreme-degree-day damage function -- and show (a) the national total stays
the same order of magnitude, (b) the spatial pattern (which counties are exposed)
is highly rank-correlated, and (c) the hedonic upper bound ($168B) uses no yield
model at all. The policy conclusion is therefore not load-bearing on the yield model.
Seed 42.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
PROJ = ROOT / "data" / "projections"
OUT = ROOT / "results" / "revision"
rp = pd.read_csv(OUT / "real_prices_2023usd.csv"); PRICE = dict(zip(rp.iloc[:, 0], rp.iloc[:, 1]))
SR = {"corn": -0.0662, "soybeans": -0.0560, "wheat_winter": -0.0420, "wheat_spring": -0.0420,
      "cotton": -0.0662, "sorghum": -0.0662, "barley": -0.0420, "oats": -0.0420}


def edd(tj, tg, thr=29.0):
    return np.maximum(0, tj - thr) * 31 + np.maximum(0, tg - thr) * 60


def county_stranded_ml(scen="SSP245", r=0.04, H=30):
    yp = pd.read_parquet(PROJ / f"yield_projections_{scen}.parquet")
    yp["fips"] = yp["fips"].astype(str).str.zfill(5)
    yp["price"] = yp["crop"].map(PRICE).fillna(4.0)
    y0 = yp["year"].min(); yp = yp[yp["year"] <= y0 + H - 1].copy()
    yp["disc"] = 1 / (1 + r) ** (yp["year"] - y0 + 1)
    yp["pv"] = yp["climate_impact_bu"].clip(upper=0) * yp["price"] * yp["acres_harvested"] * yp["disc"]
    return (-yp.groupby("fips")["pv"].sum()).rename("ml")


def county_stranded_process(scen="SSP245", climfile="county_climate_projections.parquet", r=0.04, H=30):
    yp = pd.read_parquet(PROJ / f"yield_projections_{scen}.parquet")
    yp["fips"] = yp["fips"].astype(str).str.zfill(5)
    cl = pd.read_parquet(PROJ / climfile, columns=["fips", "year", "tmax_july_projected",
         "delta_tmax_july", "tmax_growing_projected", "delta_tmax_growing"])
    cl["fips"] = cl["fips"].astype(str).str.zfill(5)
    f2c = lambda f: (f - 32) * 5 / 9
    cl["dedd"] = (edd(f2c(cl.tmax_july_projected), f2c(cl.tmax_growing_projected))
                  - edd(f2c(cl.tmax_july_projected - cl.delta_tmax_july),
                        f2c(cl.tmax_growing_projected - cl.delta_tmax_growing))).clip(lower=0)
    yp = yp.merge(cl[["fips", "year", "dedd"]], on=["fips", "year"], how="left")
    yp["dedd"] = yp["dedd"].fillna(0); yp["price"] = yp["crop"].map(PRICE).fillna(4.0)
    yp["srcoef"] = yp["crop"].map(SR).fillna(-0.0662)
    y0 = yp["year"].min(); yp = yp[yp["year"] <= y0 + H - 1].copy()
    yp["disc"] = 1 / (1 + r) ** (yp["year"] - y0 + 1)
    yp["pv"] = (yp["dedd"] * yp["srcoef"]) * yp["price"] * yp["acres_harvested"] * yp["disc"]
    return (-yp.groupby("fips")["pv"].sum()).rename("proc")


def main():
    ml = county_stranded_ml(); proc = county_stranded_process()
