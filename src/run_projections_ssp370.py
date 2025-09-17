"""Run yield projections for SSP3-7.0 using the existing v2 yield model.

Loads the SSP370 county climate projections and applies the trained LightGBM
yield model, saving results to data/projections/yield_projections_SSP370.parquet.
"""

import sys
import pickle
from pathlib import Path

