"""Audit improvement D: best combined model + a learning-curve ceiling test.

Two deliverables a hostile methods reviewer would want:

1. BEST HONEST POOLED MODEL. The drought-trajectory features (audit A) gave the
   single biggest gain (R^2 0.227 -> 0.234). Here we lock in that feature set with
   light hyper-parameter tuning and report the final pooled + per-crop held-out
   anomaly R^2 / Spearman, plus confirm the WITHIN-CROP LEVELS R^2 (the number the
   paper also reports) is not degraded.

2. LEARNING CURVE / CEILING EVIDENCE. To show the remaining gap is a genuine
   signal ceiling and not under-training, we sweep the number of training years
   (and, separately, the feature-set richness) and plot held-out anomaly R^2. If
   R^2 has plateaued -- more data and more features stop helping -- the residual
   variance is irreducible weather noise + unobserved management/irrigation, i.e.
   the literature ceiling, NOT a fixable modelling error.

Target = z-scored county-detrended anomaly. Split: test 2013-2023. Seed 42.
Climate-only features (projectable from CMIP6 deltas).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
