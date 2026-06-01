"""Corrected DCF confidence interval (Reviewer 2 #1: the [$58,63B] CI was too tight).

The honest interval must carry three error sources, not just idiosyncratic county
error. We aggregate to county present-value FIRST (fixing the stranded set, so no
per-draw rectification bias), then propagate multiplicatively:
  * idiosyncratic model error  -> independent lognormal per county (cancels in sum)
  * spatially correlated error -> one common lognormal shock per state per draw
  * GCM ensemble spread        -> per-county relative spread from p10-p90
Reported as nested CIs so the widening is transparent. Seed 42.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "revision"
np.random.seed(42)
PRICE = {"corn": 5.04, "soybeans": 12.29, "wheat_winter": 6.72, "wheat_spring": 7.38,
         "cotton": 0.93, "sorghum": 4.80, "barley": 5.64, "oats": 3.35}
