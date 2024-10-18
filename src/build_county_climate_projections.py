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
