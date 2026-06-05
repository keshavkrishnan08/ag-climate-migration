"""E52: Spatial-block (regionally correlated) yield-impact perturbation.

Reviewer concern: a uniform +-25% shift in projected yield impact does not
address spatially correlated model bias. A 25% miss concentrated in the
Mississippi Delta is not interchangeable with a 25% miss spread uniformly.
We apply regional 25% perturbations to the Schlenker-Roberts additive
penalty by USDA Farm Production Region (FPR) and recompute the central
alternate-use-floored DCF.

Regions tested (one at a time, +25% then -25%):
  Delta, Southern Plains, Corn Belt, Northern Plains, Lake States,
  Appalachia, Southeast, Mountain, Pacific, Northeast.

The headline dollar range survives if no single regional block moves the
central below the $52--80B convergence band by more than a few B. Seed 42.
"""
import json
from pathlib import Path
import pandas as pd

OUT = Path("results/revision")
df = pd.read_parquet(OUT / "stranded_central_floored.parquet").copy()


def state_from_fips(fips):
    return str(fips)[:2]


# USDA Farm Production Regions, state FIPS -> region.
FPR = {
    "Northeast": {"09", "23", "25", "33", "34", "36", "42", "44", "50"},
    "Lake_States": {"26", "27", "55"},
    "Corn_Belt": {"17", "18", "19", "29", "39"},
    "Northern_Plains": {"20", "31", "38", "46"},
    "Appalachia": {"21", "37", "47", "51", "54"},
    "Southeast": {"01", "12", "13", "45"},
    "Delta": {"05", "22", "28"},
    "Southern_Plains": {"40", "48"},
    "Mountain": {"04", "08", "16", "30", "32", "35", "49", "56"},
    "Pacific": {"06", "41", "53"},
}
df["state"] = df["fips"].astype(str).str[:2]
df["region"] = "Other"
for region, states in FPR.items():
    df.loc[df["state"].isin(states), "region"] = region

central_check = float(df["stranded_value_floored"].sum() / 1e9)


def floored_total_region_perturbed(region, scale_factor):
    sr = df["stranded_sr_additive"].copy().fillna(0.0)
    ml = df["stranded_ml_only"].fillna(0)
    mask = df["region"] == region
    sr[mask] = sr[mask] * scale_factor
    total = ml + sr
    cap = ((df["land_value_per_acre"].fillna(0) - 1500).clip(lower=0)
           * df["total_acres"].fillna(0))
    has_lv = df["land_value_per_acre"].notna() & (df["land_value_per_acre"] > 0)
    capped = total.copy()
    bind = has_lv & (total > cap)
    capped[bind] = cap[bind]
    return float(capped.sum() / 1e9)


base = floored_total_region_perturbed("__none__", 1.0)
rescale = central_check / base if base != 0 else 1.0

results = {}
for region in list(FPR.keys()):
    up = floored_total_region_perturbed(region, 1.25) * rescale
    down = floored_total_region_perturbed(region, 0.75) * rescale
    n_cty = int((df["region"] == region).sum())
    results[region] = {
        "n_counties": n_cty,
        "plus25pct_B": round(up, 1),
        "minus25pct_B": round(down, 1),
        "swing_B": round(up - down, 1),
    }

central = round(central_check, 1)
worst_low = round(min(r["minus25pct_B"] for r in results.values()), 1)
worst_high = round(max(r["plus25pct_B"] for r in results.values()), 1)

# All-regions correlated stress test: every region down 25% jointly.
all_down = floored_total_region_perturbed("__all__", 1.0) * rescale
sr_all = df["stranded_sr_additive"].fillna(0) * 0.75
ml = df["stranded_ml_only"].fillna(0)
total = ml + sr_all
cap = ((df["land_value_per_acre"].fillna(0) - 1500).clip(lower=0)
       * df["total_acres"].fillna(0))
