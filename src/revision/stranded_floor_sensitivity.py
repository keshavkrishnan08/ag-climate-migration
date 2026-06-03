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
