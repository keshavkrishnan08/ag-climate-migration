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
