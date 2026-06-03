"""Floor sensitivity: the central stranded estimate caps each county's per-acre loss at
(cropland value - alternate-use value). The central uses a $1,500/ac grazing/pasture value.
Reviewers will ask how sensitive the $61B central is to that constant. Recompute the floored
total at $1,000 and $2,000/ac. Seed 42; reads the central parquet only, writes results/revision/.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path("results/revision")
df = pd.read_parquet(OUT / "stranded_central_floored.parquet")
df = df[["fips", "stranded_before_floor", "land_value_per_acre", "total_acres", "stranded_value_floored"]].copy()
df = df.groupby("fips", as_index=False).mean(numeric_only=True)

def floored_total(pasture):
    has_lv = df["land_value_per_acre"].notna() & (df["land_value_per_acre"] > 0)
    max_loss = ((df["land_value_per_acre"] - pasture).clip(lower=0) * df["total_acres"])
    capped = df["stranded_before_floor"].copy()
    mask = has_lv & (df["stranded_before_floor"] > max_loss)
    capped[mask] = max_loss[mask]
    return float(capped.sum() / 1e9)

res = {f"pasture_{int(p)}_per_ac_central_B": round(floored_total(p), 1) for p in [1000, 1500, 2000]}
res["before_floor_central_B"] = round(float(df["stranded_before_floor"].sum() / 1e9), 1)
res["reported_floored_1500_check_B"] = round(float(df["stranded_value_floored"].sum() / 1e9), 1)
json.dump(res, open(OUT / "stranded_floor_sensitivity.json", "w"), indent=2)
print("Alternate-use floor sensitivity (central DCF, after floor):")
print("  before any floor:      $%.1fB" % res["before_floor_central_B"])
for p in [1000, 1500, 2000]:
    print("  pasture $%-5d/ac:      $%.1fB" % (p, res[f"pasture_{p}_per_ac_central_B"]))
print("  (parquet $1500 check:   $%.1fB)" % res["reported_floored_1500_check_B"])
