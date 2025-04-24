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

DPI = 300
DOUBLE_COL = 7.09   # inches
SINGLE_COL = 3.46   # inches

OUT_DIR = "/Users/keshavkrishnan/Claude/Current/AgProject/ag_migration/results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_DIR = "/Users/keshavkrishnan/Claude/Current/AgProject/ag_migration/data"

# Colour palette
C_BLUE   = "#2166AC"
C_GREEN  = "#1B7837"
C_RED    = "#D6604D"
C_ORANGE = "#F4A582"
C_YELLOW = "#FFFFBF"
C_LBLUE  = "#92C5DE"
C_LGREEN = "#A6DBA0"
C_GRAY   = "#636363"
C_DARK   = "#252525"


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE A — Methods Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def fig_methods_pipeline():
    fig = plt.figure(figsize=(DOUBLE_COL, 4.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(ax, x, y, w, h, label, sublabel="", color="#FFFFFF", edgecolor=C_BLUE,
            fontsize=6.5, bold=False):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle="round,pad=0.08",
                               facecolor=color, edgecolor=edgecolor,
                               linewidth=0.8, zorder=3)
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x, y + (0.12 if sublabel else 0), label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=C_DARK, zorder=4)
        if sublabel:
            ax.text(x, y - 0.2, sublabel, ha="center", va="center",
                    fontsize=5.5, color=C_GRAY, zorder=4, style="italic")

    def arrow(ax, x1, y1, x2, y2, color=C_GRAY, lw=0.8, style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color,
                                   lw=lw, connectionstyle="arc3,rad=0"))

    # ── Section headers ──
    ax.text(1.4, 8.6, "DATA INPUTS", ha="center", va="center",
            fontsize=7, fontweight="bold", color=C_BLUE)
    ax.text(5.0, 8.6, "ANALYTICAL PIPELINE", ha="center", va="center",
            fontsize=7, fontweight="bold", color=C_GREEN)
    ax.text(8.7, 8.6, "FINDINGS", ha="center", va="center",
            fontsize=7, fontweight="bold", color=C_RED)

    ax.axvline(2.9, 0.02, 0.97, color="#DDDDDD", lw=0.5, zorder=0)
