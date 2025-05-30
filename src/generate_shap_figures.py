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
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = "/Users/keshavkrishnan/Claude/Current/AgProject/ag_migration"
MODEL_PATH       = os.path.join(BASE_DIR, "results/20260317_192605/yield_model.pkl")
FEATURE_MATRIX   = os.path.join(BASE_DIR, "data/processed/feature_matrix.parquet")
PROJECTIONS_PATH = os.path.join(BASE_DIR, "data/projections/yield_projections_SSP245.parquet")
OUT_DIR          = os.path.join(BASE_DIR, "results/figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Nature Food style constants
# ---------------------------------------------------------------------------
DOUBLE_COL_IN = 7.09        # 180 mm
SINGLE_COL_IN = 3.43        # 87  mm
DPI           = 300
FONTSIZE_BASE = 7
FONTSIZE_TICK = 6
FONTSIZE_LABEL = 7

NATURE_RC = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         FONTSIZE_BASE,
    "axes.labelsize":    FONTSIZE_LABEL,
    "axes.titlesize":    FONTSIZE_LABEL,
    "xtick.labelsize":   FONTSIZE_TICK,
    "ytick.labelsize":   FONTSIZE_TICK,
    "legend.fontsize":   FONTSIZE_TICK,
    "lines.linewidth":   0.8,
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  2.0,
    "ytick.major.size":  2.0,
    "pdf.fonttype":      42,    # TrueType in PDF
    "ps.fonttype":       42,
}
plt.rcParams.update(NATURE_RC)

# Colorblind-safe palette (Wong 2011)
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"
PURPLE = "#CC79A7"
LBLUE  = "#56B4E9"
YELLOW = "#F0E442"
BLACK  = "#000000"

REGION_COLORS = {
    "Corn Belt":       BLUE,
    "Southern Plains": ORANGE,
    "Northern Plains": GREEN,
    "Southeast":       RED,
}

# State FIPS for each region
REGION_STATE_FIPS = {
    "Corn Belt":       ["17", "18", "19", "39"],   # IL, IN, IA, OH
    "Southern Plains": ["48", "40", "20"],           # TX, OK, KS
    "Northern Plains": ["38", "46", "31", "27"],    # ND, SD, NE, MN
    "Southeast":       ["13", "01", "28", "45", "37"],  # GA, AL, MS, SC, NC
}

# Model features in order
MODEL_FEATURES = [
    "yield_trend_slope_15yr", "yield_trend_intercept",
    "tmax_july_c", "tmax_growing_c", "tmin_growing_c",
    "precip_growing", "pdsi_growing", "cdd_annual",
    "gdd_corn", "gdd_soybeans", "gdd_wheat_winter", "gdd_wheat_spring",
    "gdd_cotton", "gdd_sorghum", "gdd_barley", "gdd_oats",
    "tmax_july_c_trend10", "precip_growing_trend10", "cdd_annual_trend10",
    "tmax_july_c_anomaly", "precip_growing_anomaly", "pdsi_growing_anomaly",
    "extreme_heat_months", "switching_rate_proxy", "switching_rate_5yr",
    "log_population", "log_median_income", "poverty_rate",
    "crop_barley", "crop_corn", "crop_cotton", "crop_oats",
    "crop_sorghum", "crop_soybeans", "crop_wheat_spring", "crop_wheat_winter",
]
CROP_DUMMIES = [f for f in MODEL_FEATURES if f.startswith("crop_")]
CROP_NAMES   = [c.replace("crop_", "") for c in CROP_DUMMIES]


# ---------------------------------------------------------------------------
# Helper: one-hot encode crop column
# ---------------------------------------------------------------------------
def add_crop_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary crop dummy columns to match the 36-feature model schema.

    Args:
