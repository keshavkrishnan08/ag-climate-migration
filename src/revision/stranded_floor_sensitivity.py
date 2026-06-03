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
