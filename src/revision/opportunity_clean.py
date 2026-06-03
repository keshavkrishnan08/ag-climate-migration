"""Clean opportunity recompute with a real cropland denominator (removes the
all-farm-acreage inflation, e.g. orchards in Yakima).

Expandable acreage is capped at the county's historical MAXIMUM total harvested
acreage (land that has actually been cropped) plus a 15% idle-cropland margin,
instead of all-farm 'ACRES OPERATED' x a state fraction. Opportunity is reported
as NET farm income (22% USDA-ERS grain/oilseed operating margin). Seed 42.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "revision"
PRICE = {"corn": 5.04, "soybeans": 12.29, "wheat_winter": 6.72, "wheat_spring": 7.38,
         "cotton": 0.93, "sorghum": 4.80, "barley": 5.64, "oats": 3.35}
MIN_VIABLE = {"corn": 60, "soybeans": 20, "wheat_winter": 30, "wheat_spring": 20,
              "barley": 30, "oats": 25, "sorghum": 40}
MARGIN = 0.22
EXPANSION_HEADROOM = 1.15   # allow 15% onto historically idle cropland


