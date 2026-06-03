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
