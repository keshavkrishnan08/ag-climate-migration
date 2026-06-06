"""Fair comparison: same %-deviation target, OLD growing-season aggregate features
(no temperature spectrum). Isolates how much the spectrum representation adds vs
the original aggregates, holding the target fixed.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
