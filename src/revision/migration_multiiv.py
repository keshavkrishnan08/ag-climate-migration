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
