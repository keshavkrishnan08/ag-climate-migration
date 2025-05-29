"""
Generate SHAP analysis figures for the yield model.

Produces:
  - fig03_yield_cliff.pdf/.png  — SHAP dependence plots (tmax_july_c vs corn,
                                   precip_growing vs soybeans)
  - fig11_uncertainty.pdf/.png  — Regional yield projection trajectories with
                                   10th-90th percentile uncertainty bands

Nature Food formatting: Arial 7pt, 300 DPI, double-column 180mm = 7.09 in.

Args (via CLI or direct run):
    model_path: path to yield_model.pkl
    feature_matrix_path: path to feature_matrix.parquet
    projections_path: path to yield_projections_SSP245.parquet
    output_dir: directory to write figures

Returns:
    Saves PDF + PNG for each figure; prints SHAP summary stats to stdout.

Raises:
    FileNotFoundError: if any input path does not exist.
    ValueError: if expected features are missing from feature matrix.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import shap

# ---------------------------------------------------------------------------
