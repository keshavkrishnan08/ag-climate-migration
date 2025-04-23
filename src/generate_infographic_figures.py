"""
generate_infographic_figures.py
--------------------------------
Six publication-quality infographic/diagram figures for the AgMigration paper.
Nature Food style: Arial 7pt min, 300 DPI.
  - Double column = 180 mm = 7.09 in
  - Single column = 88 mm = 3.46 in

Figures generated:
  A. fig_methods_pipeline.pdf  — full analytical pipeline flowchart
  B. fig_northward_summary.pdf — centroid shift summary (map + bar chart)
  C. fig_temp_response.pdf     — temperature-yield response curves
  D. fig_cascade_mechanism.pdf — 7-step cascade flow diagram
  E. fig_insurance_flow.pdf    — insurance cross-subsidy flow visual
  F. fig_valuation_methods.pdf — three-method valuation comparison
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, ArrowStyle
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

# ── Typography & style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
