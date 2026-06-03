"""Revision: identification-robust migration IV (Reviewer 1 #2).

The original weather IV was criticised on two grounds: (i) external validity
(the yield->migration link should hold only where farming is a large income
share) and (ii) the exclusion restriction (local heat/drought can move people
through amenity or winter-mildness channels, not only farm income; Rappaport
2007). This rebuild answers both:

  * EXTERNAL VALIDITY: estimate only on ERS farming-dependent counties
    (Type_2015_Farming_NO = 1).

  * EXCLUSION RESTRICTION: instrument county farm-income deviation with a
    LEAVE-ONE-OUT SHIFT-SHARE (Bartik) shock built from OTHER counties' national
    crop-specific yield shocks weighted by the county's pre-period crop mix:
        z_it = sum_c  share_ic(baseline) * price_c * g_{c,t}^{-i}
    where g_{c,t}^{-i} is the mean detrended yield anomaly of crop c in year t
    across all counties EXCEPT i. Because the instrument uses other counties'
    growing conditions, it does not carry county i's own local weather, so it
    cannot move county i's migration through a local-amenity channel. We
    additionally control for the county's own winter-minimum-temperature anomaly
    and an amenity indicator, and run an amenity PLACEBO: the same instrument has
    no reduced-form effect on migration in non-farming counties.

  * SUSTAINED SHOCK: treatment is the 3-year moving average of farm-income
    deviation (out-migration responds to persistent decline, not single years).

Seed 42. Writes only to results/revision/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

PRICE = {"corn": 5.04, "soybeans": 12.29, "wheat_winter": 6.72, "wheat_spring": 7.38,
         "cotton": 0.93, "sorghum": 4.80, "barley": 5.64, "oats": 3.35}


def farming_dependent():
    cc = pd.read_csv(DATA_RAW / "other" / "ers_atlas" / "CountyClassifications.csv",
                     dtype=str, encoding="latin-1")
    cc = cc.rename(columns={cc.columns[0]: "fips"})
    cc["fips"] = cc["fips"].str.zfill(5)
    fd = cc[cc["Attribute"] == "Type_2015_Farming_NO"][["fips", "Value"]].copy()
    fd["farm_dependent"] = (fd["Value"] == "1").astype(int)
    am = cc[cc["Attribute"] == "HiAmenity"][["fips", "Value"]].rename(
        columns={"Value": "hi_amenity"})
    am["hi_amenity"] = pd.to_numeric(am["hi_amenity"], errors="coerce").fillna(0)
    return fd[["fips", "farm_dependent"]].merge(am, on="fips", how="left")


def winter_temp_anomaly():
    m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet",
                        columns=["fips", "year", "tmin_m12", "tmin_m01", "tmin_m02"])
    m["fips"] = m["fips"].astype(str).str.zfill(5)
    m["winter_tmin"] = m[["tmin_m12", "tmin_m01", "tmin_m02"]].mean(axis=1)
    m["winter_tmin_anom"] = m["winter_tmin"] - m.groupby("fips")["winter_tmin"].transform("mean")
    return m[["fips", "year", "winter_tmin_anom"]]


def build_panel():
    fm = pd.read_parquet(DATA_PROCESSED / "feature_matrix.parquet",
                         columns=["fips", "year", "crop", "yield_bu_acre",
                                  "yield_anomaly", "acres_harvested"])
    fm["fips"] = fm["fips"].astype(str).str.zfill(5)
    fm["price"] = fm["crop"].map(PRICE).fillna(5.0)
    fm["revenue"] = fm["yield_bu_acre"] * fm["acres_harvested"].clip(lower=0) * fm["price"]

    # County farm revenue per year (2023 USD proxy)
    rev = fm.groupby(["fips", "year"])["revenue"].sum().reset_index()
    base = rev[rev["year"].between(2000, 2009)].groupby("fips")["revenue"].mean().rename("base_rev")
    rev = rev.merge(base, on="fips", how="left")
    rev = rev[rev["base_rev"] > 0]
    rev["farm_income_dev"] = (rev["revenue"] - rev["base_rev"]) / rev["base_rev"]
    rev = rev.sort_values(["fips", "year"])
    rev["fid_3yr"] = rev.groupby("fips")["farm_income_dev"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())

    # Baseline crop mix (2000-2009 acreage shares)
    bmix = fm[fm["year"].between(2000, 2009)].groupby(["fips", "crop"])["acres_harvested"].mean()
    bmix = bmix.reset_index()
    tot = bmix.groupby("fips")["acres_harvested"].transform("sum")
    bmix["share"] = bmix["acres_harvested"] / tot.replace(0, np.nan)
    bmix["price"] = bmix["crop"].map(PRICE).fillna(5.0)

    # Leave-one-out national crop yield shock g_{c,t}^{-i}
    # national sum and count of yield_anomaly by crop-year, then loo per county
    natl = fm.groupby(["crop", "year"])["yield_anomaly"].agg(["sum", "count"]).reset_index()
    fm2 = fm.merge(natl, on=["crop", "year"], how="left")
    fm2["g_loo"] = (fm2["sum"] - fm2["yield_anomaly"]) / (fm2["count"] - 1).clip(lower=1)
    # Bartik: sum_c share_ic * price_c * g_loo_{c,t}
    fm2 = fm2.merge(bmix[["fips", "crop", "share"]], on=["fips", "crop"], how="left")
    fm2["bartik_term"] = fm2["share"].fillna(0) * fm2["price"] * fm2["g_loo"]
    bartik = fm2.groupby(["fips", "year"])["bartik_term"].sum().rename("z_bartik").reset_index()

    panel = (rev.merge(bartik, on=["fips", "year"], how="inner")
               .merge(farming_dependent(), on="fips", how="left")
               .merge(winter_temp_anomaly(), on=["fips", "year"], how="left"))
    panel["farm_dependent"] = panel["farm_dependent"].fillna(0).astype(int)
    panel["hi_amenity"] = panel["hi_amenity"].fillna(0)

    # Outcomes from ACS migration + demographics
    demo = pd.read_parquet(DATA_RAW / "census" / "acs_county_demographics.parquet",
                           columns=["fips", "year", "total_population"])
    demo["fips"] = demo["fips"].astype(str).str.zfill(5)
    demo = demo.sort_values(["fips", "year"])
    # 3-year forward population growth (cumulative) — less noisy than 1-yr
    demo["pop_lead3"] = demo.groupby("fips")["total_population"].shift(-3)
    demo["pop_growth_3yr"] = (demo["pop_lead3"] - demo["total_population"]) / demo["total_population"]

    mig = pd.read_parquet(DATA_RAW / "census" / "acs_migration_data.parquet")
    mig["fips"] = mig["fips"].astype(str).str.zfill(5)
    mig = mig.merge(demo[["fips", "year", "total_population"]], on=["fips", "year"], how="left")
    mig["in_mig_rate"] = mig["moved_diff_county_same_state"].fillna(0) / mig["total_population"]

    panel = (panel.merge(demo[["fips", "year", "pop_growth_3yr", "total_population"]],
                         on=["fips", "year"], how="left")
                  .merge(mig[["fips", "year", "in_mig_rate"]], on=["fips", "year"], how="left"))
    return panel


def demean2(df, cols, ent="fips", time="year"):
    out = df.copy()
    for c in cols:
        s = out[c].astype(float)
        for _ in range(25):
            s = s - s.groupby(out[ent]).transform("mean")
            s = s - s.groupby(out[time]).transform("mean")
        out[c + "_dm"] = s
    return out


def tsls(df, y, d, z, controls, cluster="fips"):
    """Two-way FE 2SLS with cluster-robust SE on the endogenous coefficient."""
    cols = [y, d, z] + controls
    dd = df.dropna(subset=cols).copy()
    dd = dd[np.all(np.isfinite(dd[cols].values), axis=1)]
    if dd["fips"].nunique() < 20 or len(dd) < 100:
        return None
    dm = demean2(dd, cols)
