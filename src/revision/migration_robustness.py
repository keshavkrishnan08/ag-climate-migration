"""Migration robustness: weak-IV-robust Anderson-Rubin confidence set,
alternative outcomes, and leave-one-crop-out shift-share stability.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from migration_iv_bartik import build_panel, demean2
OUT = ROOT / "results" / "revision"


def high_tercile(panel):
    base = panel.groupby("fips").agg(base_rev=("base_rev", "first")).reset_index()
    pop0 = panel.sort_values("year").groupby("fips")["total_population"].first().rename("pop0").reset_index()
    fi = base.merge(pop0, on="fips"); fi["fi"] = fi["base_rev"] / fi["pop0"].replace(0, np.nan)
    panel = panel.merge(fi[["fips", "fi"]], on="fips", how="left")
    return panel[panel["fi"] >= panel["fi"].quantile(0.67)].copy()
