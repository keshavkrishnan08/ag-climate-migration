"""
Build county-level climate projections from CMIP6 gridded data using the delta method.

Units:
  PRISM baseline  : Fahrenheit (tmax, tmin), mm/month (precip)
  CMIP6 tasmax/min: Kelvin  → convert delta to °F before adding to PRISM
  CMIP6 pr        : kg/m²/s → convert to mm/month, then take delta

Delta method:
  delta = GCM(target_year) - GCM(2025-2030_mean)
  projected = PRISM_1981-2010_baseline + delta

Workflow:
  1. Load county centroids (CONUS only)
  2. PRISM 1981-2010 baseline per county
  3. Build CMIP6 grid → nearest county lookup
  4. Load GCM data: reference period (2025-2030) + 5 rep years (2030,35,40,45,50)
  5. Delta method + ensemble stats at each rep year
  6. Linear interpolation to annual 2025-2050
  7. Save parquet + print summary
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
CMIP6_DIR   = BASE / "data/raw/cmip6"
PRISM_PATH  = BASE / "data/raw/prism/county_climate_monthly.parquet"
GAZETTE_PATH= BASE / "data/raw/census/2023_Gaz_counties_national.txt"
OUT_PATH    = BASE / "data/projections/county_climate_projections.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
GCMS            = [
    # Original 5 (r1i1p1f1, standard calendar)
    "ACCESS-CM2", "GFDL-ESM4", "MIROC6", "MPI-ESM1-2-HR", "NorESM2-MM",
    # New 5 (Pangeo Cloud via anonymous GCS zarr)
    # CESM2: r10i1p1f1 (no r1i1p1f1 for ssp245 in Pangeo)
    # CNRM-CM6-1: r1i1p1f2 (uses f2 physics variant)
    # HadGEM3-GC31-LL: r1i1p1f3 (AAER forcing; 360-day calendar handled at download)
    # IPSL-CM6A-LR: r1i1p1f1
    # MRI-ESM2-0: r1i1p1f1
    "CESM2", "CNRM-CM6-1", "HadGEM3-GC31-LL", "IPSL-CM6A-LR", "MRI-ESM2-0",
]
REP_YEARS       = [2030, 2035, 2040, 2045, 2050]
REF_YEARS       = list(range(2025, 2031))   # 2025–2030 GCM reference
GROW_MONTHS     = [5, 6, 7, 8, 9]           # May–Sep
