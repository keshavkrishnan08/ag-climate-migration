"""
Fix download for models using non-standard calendars (CESM2 and HadGEM3-GC31-LL).

CESM2 uses DatetimeNoLeap; HadGEM3-GC31-LL uses Datetime360Day.
Both require using cftime objects directly rather than converting to pandas DatetimeIndex.

Outputs same parquet format as download_cmip6_pangeo.py.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs
import cftime
from pathlib import Path
import time

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent.parent
