"""Prime-age (25-54) population in the WITHIN-COUNTY panel-FE shift-share IV — the
design that holds for total population (the long-difference fails only because it is
a cross-section with no fixed effects). Outcome: 3-yr-forward prime-age growth.
County+year FE; instrument = leave-one-out shift-share. High-farm-intensity tercile
of farming-dependent counties + placebo. Seed 42."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(ROOT/"src"/"revision"))
from migration_iv_bartik import build_panel
OUT=ROOT/"results"/"revision"; np.random.seed(42)

def within(df,cols,gi="fips",gt="year"):
    o=df.copy()
    for c in cols:
        s=o[c].astype(float)
        for _ in range(30): s=s-s.groupby(o[gi]).transform("mean"); s=s-s.groupby(o[gt]).transform("mean")
        o[c+"_w"]=s
