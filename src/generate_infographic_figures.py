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
    ax.axvline(7.1, 0.02, 0.97, color="#DDDDDD", lw=0.5, zorder=0)

    # ── Data input boxes (left column) ──
    data_inputs = [
        ("NASS", "638K obs\n1950–2023", 7.9),
        ("PRISM", "4 km daily\nclimate grid", 6.7),
        ("CMIP6", "10 GCMs\n4 scenarios", 5.5),
        ("RMA/FCIC", "Insurance\npremiums & losses", 4.3),
        ("ACS/Census", "Demographics\n2902 counties", 3.1),
        ("USDA CDL", "Land cover\n2008–2023", 1.9),
    ]
    for label, sub, y in data_inputs:
        box(ax, 1.4, y, 2.4, 0.7, label, sub, color="#E8F4FD", edgecolor=C_BLUE)

    # ── Pipeline center boxes ──
    pipe_steps = [
        (5.0, 7.9, "Feature Engineering",  "638,808 obs × 34 features",   "#E8F5E9", C_GREEN),
        (5.0, 6.5, "County Yield Model",   "Fixed-FX + climate quadratics","#E8F5E9", C_GREEN),
        (5.0, 5.1, "Crop Switching Model", "Multinomial logit, 2902 FIPS", "#E8F5E9", C_GREEN),
        (5.0, 3.7, "GCM Projections",      "10 GCMs × 4 SSPs, 2025–2100",  "#E8F5E9", C_GREEN),
        (5.0, 2.3, "Impact Quantification","Monte Carlo, n=1000",           "#E8F5E9", C_GREEN),
    ]
    for x, y, label, sub, col, edge in pipe_steps:
        box(ax, x, y, 3.4, 0.75, label, sub, color=col, edgecolor=edge, bold=True)

    # Arrows between pipeline stages
    for y_top, y_bot in [(7.53, 6.87), (6.13, 5.47), (4.73, 4.07), (3.33, 2.67)]:
        arrow(ax, 5.0, y_top, 5.0, y_bot, color=C_GREEN, lw=1.2)

    # Arrows from data to pipeline
    data_y   = [7.9, 6.7, 5.5, 4.3, 3.1, 1.9]
    pipe_y   = [7.9, 6.5, 5.1, 3.7, 3.7, 2.3]
    for dy, py in zip(data_y, pipe_y):
        arrow(ax, 2.65, dy, 3.3, py, color="#9ECAE1", lw=0.6, style="-|>")

    # ── Findings output boxes (right column) ──
    findings = [
        ("Stranded Assets",   "$56–168B\nland value at risk",   7.4, C_RED),
        ("Cascade Effects",   "337 counties\ntipping by 2040",  5.9, C_RED),
        ("Insurance Mispricing", "$5.9B/yr\ncross-subsidy",     4.4, C_RED),
        ("Opportunity Zones", "$51B\nnorthern gains",           2.9, "#7B2D8B"),
    ]
    for label, sub, y, edge in findings:
        box(ax, 8.7, y, 2.3, 0.82, label, sub, color="#FEE5D9", edgecolor=edge)

    # Arrows from impact quantification to findings
    for fy in [7.4, 5.9, 4.4, 2.9]:
        arrow(ax, 6.7, 2.3, 7.55, fy, color=C_RED, lw=0.6, style="-|>")

    # ── N-count annotations ──
    ax.text(5.0, 1.5, "n = 638,808 obs  |  2,902 counties  |  10 GCMs  |  4 SSPs  |  1,000 Monte Carlo draws",
            ha="center", va="center", fontsize=5.5, color=C_GRAY,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F7F7F7", edgecolor="#CCCCCC", lw=0.5))

    ax.text(5.0, 0.6, "Figure A — Full Analytical Pipeline",
            ha="center", va="center", fontsize=6, color=C_GRAY, style="italic")

    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUT_DIR}/fig_methods_pipeline.{ext}", dpi=DPI)
    plt.close(fig)
    print("  [A] fig_methods_pipeline — done")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE B — Northward Shift Summary
# ══════════════════════════════════════════════════════════════════════════════
def fig_northward_summary():
    # Load data
    yields = pd.read_parquet(f"{DATA_DIR}/raw/nass/nass_county_yields.parquet")
    gaz    = pd.read_csv(f"{DATA_DIR}/raw/census/2023_Gaz_counties_national.txt",
                         sep="\t", encoding="latin-1")

    gaz.columns = gaz.columns.str.strip()
    gaz["GEOID"] = gaz["GEOID"].astype(str).str.zfill(5)
    yields["fips"] = yields["fips"].astype(str).str.zfill(5)

    lat_col  = "INTPTLAT"
    lon_col  = [c for c in gaz.columns if "INTPTLONG" in c][0]
    gaz      = gaz[["GEOID", lat_col, lon_col]].rename(
        columns={"GEOID": "fips", lat_col: "lat", lon_col: "lon"})
    gaz["lon"] = pd.to_numeric(gaz["lon"], errors="coerce")
    gaz["lat"] = pd.to_numeric(gaz["lat"], errors="coerce")

    merged = yields.merge(gaz, on="fips", how="inner")
    merged = merged[merged["acres_harvested"] > 0]

    crops_of_interest = {
        "corn":         ("#E6550D", "Corn"),
        "soybeans":     ("#31A354", "Soybeans"),
        "wheat_winter": ("#756BB1", "Winter Wheat"),
        "cotton":       ("#3182BD", "Cotton"),
    }

    early_yr  = (1960, 1970)
    recent_yr = (2013, 2023)

    centroids = {}
    rates     = {}
    for crop in crops_of_interest:
        sub = merged[merged["crop"] == crop].copy()
        early  = sub[sub["year"].between(*early_yr)].groupby("fips").agg(
            acres=("acres_harvested", "mean"), lat=("lat", "first"), lon=("lon", "first")).reset_index()
        recent = sub[sub["year"].between(*recent_yr)].groupby("fips").agg(
            acres=("acres_harvested", "mean"), lat=("lat", "first"), lon=("lon", "first")).reset_index()

        def wcent(df):
            w = df["acres"].clip(lower=0)
            if w.sum() == 0:
                return df["lat"].mean(), df["lon"].mean()
            return (df["lat"] * w).sum() / w.sum(), (df["lon"] * w).sum() / w.sum()

        if len(early) > 10 and len(recent) > 10:
            lat_e, lon_e = wcent(early)
            lat_r, lon_r = wcent(recent)
            centroids[crop] = (lat_e, lon_e, lat_r, lon_r)
            # miles/decade: ~69 miles per degree lat; span ~60 years = 6 decades
            shift_deg = lat_r - lat_e
            shift_mi  = shift_deg * 69.0
            n_decades = (np.mean(recent_yr) - np.mean(early_yr)) / 10.0
            rates[crop] = shift_mi / n_decades if n_decades > 0 else 0.0

    # ── Figure layout ──
    fig = plt.figure(figsize=(DOUBLE_COL, 3.8))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1], wspace=0.35,
                   left=0.06, right=0.97, top=0.88, bottom=0.15)

    ax_map = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # ── Simple CONUS bounding box map ──
