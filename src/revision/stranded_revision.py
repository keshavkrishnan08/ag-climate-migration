"""Revision script: stranded farmland DCF recomputation with defensible real prices.

Responds to Reviewer 1, Major #1:
  (a) Uses 30-yr (1994-2023) inflation-adjusted marketing-year average prices
      instead of near-peak nominal prices. Prices held flat in real terms —
      consistent with USDA long-run price outlook (USDA 2024a).
  (b) Adds alternate-use floor: per-acre losses capped at
      (cropland value - pasture/grazing land value) where cropping returns
      go negative.

CONVENTIONS:
  - All dollars in 2023 USD.
  - FIPS are 5-digit zero-padded strings.
  - Random seed 42 (no randomness here, but noted for reproducibility).

Author: AgMigration revision team, 2026-05-21
"""

import sys
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
RESULTS_REV = PROJECT_ROOT / "results" / "revision"

