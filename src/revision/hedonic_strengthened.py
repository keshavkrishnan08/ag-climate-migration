"""Strengthen the hedonic against the Ricardian omitted-variable critique
(Deschenes & Greenstone 2007) by controlling for the exact confounders it names:
SSURGO soil available-water capacity, county irrigation share, and a soil-
productivity index. If the warming (temperature) coefficient is stable when these
are added, the cross-sectional bias concern is materially defused.

Also: (a) recompute the hedonic stranded value with the full-control model and a
spatially clustered (state) bootstrap CI; (b) measure the field-crop share of
agricultural cash receipts to reconcile the DCF (field crops) with the hedonic
(all channels) using a measured number rather than an assumed 30%.

Seed 42; writes only to results/revision/.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
