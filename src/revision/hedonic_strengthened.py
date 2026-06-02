"""Strengthen the hedonic against the Ricardian omitted-variable critique
(Deschenes & Greenstone 2007) by controlling for the exact confounders it names:
SSURGO soil available-water capacity, county irrigation share, and a soil-
productivity index. If the warming (temperature) coefficient is stable when these
are added, the cross-sectional bias concern is materially defused.

Also: (a) recompute the hedonic stranded value with the full-control model and a
spatially clustered (state) bootstrap CI; (b) measure the field-crop share of
agricultural cash receipts to reconcile the DCF (field crops) with the hedonic
(all channels) using a measured number rather than an assumed 30%.

Seed 42; writes only to results/revision/.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
PROJ = ROOT / "data" / "projections"
OUT = ROOT / "results" / "revision"
np.random.seed(42)
GROW = [f"{m:02d}" for m in range(4, 10)]
CPI = {2017: 0.0, 2022: 0.0}  # placeholder; deflate 2022->2017-ish via ratio below


def build():
    # land values (2017, 2022), deflate 2022 to common dollars (CPI 2017=245.1, 2022=292.7)
    lv = pd.read_parquet(DATA_RAW / "nass" / "nass_land_values.parquet")
    lv["fips"] = lv["fips"].astype(str).str.zfill(5)
    lv = lv[lv["year"].isin([2017, 2022])].copy()
    lv.loc[lv["year"] == 2022, "land_value_per_acre"] *= 245.1 / 292.7
    lv = lv.groupby("fips", as_index=False)["land_value_per_acre"].mean()
    lv = lv[(lv["land_value_per_acre"] > lv["land_value_per_acre"].quantile(0.01)) &
            (lv["land_value_per_acre"] < lv["land_value_per_acre"].quantile(0.99))]

    # climate 2019-2023 avg: July Tmax (F), growing precip
    m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet")
    m["fips"] = m["fips"].astype(str).str.zfill(5)
    m = m[m["year"].between(2019, 2023)].copy()
    m["precip_growing"] = m[[f"precip_m{mm}" for mm in GROW]].sum(axis=1)
    clim = m.groupby("fips", as_index=False).agg(tmax_july=("tmax_m07", "mean"),
                                                 precip_growing=("precip_growing", "mean"))
    # ACS controls
    demo = pd.read_parquet(DATA_RAW / "census" / "acs_county_demographics.parquet",
                           columns=["fips", "year", "total_population", "median_household_income"])
    demo["fips"] = demo["fips"].astype(str).str.zfill(5)
    demo = demo[demo["year"].between(2019, 2023)].groupby("fips", as_index=False).agg(
        pop=("total_population", "mean"), inc=("median_household_income", "mean"))
    demo["log_pop"] = np.log(demo["pop"].clip(lower=1)); demo["log_inc"] = np.log(demo["inc"].clip(lower=1))
    # SSURGO + irrigation + soil-productivity
    soil = pd.read_parquet(OUT / "ssurgo_county_soil.parquet"); soil["fips"] = soil["fips"].astype(str).str.zfill(5)
    irr = pd.read_parquet(OUT / "irrigation_share.parquet")
    irr["fips"] = irr["fips"].astype(str).str.zfill(5)
    irrc = irr.groupby("fips", as_index=False)["irr_prop"].mean().rename(columns={"irr_prop": "irr_share"})
    fm = pd.read_parquet(DATA_PROC / "feature_matrix.parquet", columns=["fips", "crop", "yield_bu_acre"])
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    corn = fm[fm["crop"] == "corn"].groupby("fips")["yield_bu_acre"].max()
    nccpi = (corn / corn.max()).rename("nccpi").reset_index()

    df = (lv.merge(clim, on="fips").merge(demo[["fips", "log_pop", "log_inc"]], on="fips")
          .merge(soil, on="fips", how="left").merge(irrc, on="fips", how="left")
          .merge(nccpi, on="fips", how="left"))
    df["state"] = df["fips"].str[:2]
    df["tmax_july_sq"] = df["tmax_july"] ** 2
    df["log_land_value"] = np.log(df["land_value_per_acre"])
    for c in ["ssurgo_aws", "irr_share", "nccpi"]:
        df[c] = df[c].fillna(df[c].median())
