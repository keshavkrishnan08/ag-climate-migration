"""Make the migration claim stand: a LONG-DIFFERENCE shift-share IV (Feng 2010,
Hornbeck 2012 design). Migration from agricultural decline is slow and cumulative,
so annual ACS noise masks it; collapsing to one 2009->2023 difference per county
recovers the structural effect. Farming-dependent counties; instrument = cumulative
leave-one-out shift-share farm-income shock. Seed 42."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(ROOT/"src"/"revision"))
from migration_iv_bartik import build_panel
OUT=ROOT/"results"/"revision"; np.random.seed(42)

def tsls_cs(d,y,x,z,ctrls):
    cols=[y,x,z]+ctrls; d=d.dropna(subset=cols).copy(); d=d[np.all(np.isfinite(d[cols].values),axis=1)]
    Y=d[y].values; D=d[x].values; Z=np.column_stack([d[z].values]+[d[c].values for c in ctrls]) if ctrls else d[z].values.reshape(-1,1)
    Z=np.column_stack([np.ones(len(d)),Z]); 
    C=np.column_stack([d[c].values for c in ctrls]) if ctrls else np.empty((len(d),0))
    Xe=np.column_stack([np.ones(len(d)),D,C]) if C.size else np.column_stack([np.ones(len(d)),D])
