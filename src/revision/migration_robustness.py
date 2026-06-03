"""Migration robustness: weak-IV-robust Anderson-Rubin confidence set,
alternative outcomes, and leave-one-crop-out shift-share stability.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
