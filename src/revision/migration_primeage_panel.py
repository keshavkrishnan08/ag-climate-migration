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
