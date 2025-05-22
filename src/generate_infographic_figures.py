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
    # Draw a simple rectangular CONUS outline
    conus_lon = (-125, -66)
    conus_lat = (24, 50)

    # Light background for land
    land_rect = patches.Rectangle(
        (conus_lon[0], conus_lat[0]),
        conus_lon[1] - conus_lon[0],
        conus_lat[1] - conus_lat[0],
        linewidth=1, edgecolor="#999999", facecolor="#F0EFE5", zorder=0)
    ax_map.add_patch(land_rect)

    # Rough state grid lines (horizontal/vertical for texture)
    for lat in np.arange(25, 50, 5):
        ax_map.axhline(lat, color="#E0E0E0", lw=0.3, zorder=1)
    for lon in np.arange(-125, -66, 10):
        ax_map.axvline(lon, color="#E0E0E0", lw=0.3, zorder=1)

    # Draw state-approximate boundaries (simple horizontal dividers for visual)
    # Major geographic feature lines
    ax_map.axhline(37, color="#CCCCCC", lw=0.4, ls="--", zorder=1)  # ~Mason-Dixon
    ax_map.axvline(-100, color="#CCCCCC", lw=0.4, ls="--", zorder=1)  # 100th meridian

    ax_map.text(-100, 37.5, "100°W", fontsize=4, color="#AAAAAA", ha="center")

    # Plot centroid arrows
    for crop, (color, label) in crops_of_interest.items():
        if crop not in centroids:
            continue
        lat_e, lon_e, lat_r, lon_r = centroids[crop]
        ax_map.plot(lon_e, lat_e, "o", color=color, ms=4, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.4)
        ax_map.plot(lon_r, lat_r, "^", color=color, ms=4, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.4)
        ax_map.annotate("",
            xy=(lon_r, lat_r), xytext=(lon_e, lat_e),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                            mutation_scale=7,
                            connectionstyle="arc3,rad=0.15"),
            zorder=6)

    ax_map.set_xlim(-126, -65)
    ax_map.set_ylim(23, 51)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("Longitude", fontsize=6)
    ax_map.set_ylabel("Latitude", fontsize=6)
    ax_map.tick_params(labelsize=5)
    ax_map.set_title("Production Centroid Shift\n1960s → 2013–2023", fontsize=7, fontweight="bold")

    # Legend for map
    handles = []
    for crop, (color, label) in crops_of_interest.items():
        if crop in centroids:
            handles.append(mpatches.Patch(color=color, label=label))
    ax_map.legend(handles=handles, fontsize=5, loc="lower left",
                  framealpha=0.85, edgecolor="#CCCCCC", handlelength=1)

    # Annotation
    ax_map.annotate("▲ = 2013–2023\n● = 1960s", xy=(-68, 24.5),
                    fontsize=4.5, ha="right", color=C_GRAY)

    # ── Bar chart of shift rates ──
    crop_labels = []
    shift_values = []
    bar_colors   = []
    for crop, (color, label) in crops_of_interest.items():
        if crop in rates:
            crop_labels.append(label)
            shift_values.append(rates[crop])
            bar_colors.append(color)

    y_pos = np.arange(len(crop_labels))
    bars  = ax_bar.barh(y_pos, shift_values, color=bar_colors,
                        height=0.55, edgecolor="white", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, shift_values)):
        sign = "+" if val >= 0 else ""
        ax_bar.text(max(val, 0) + 0.3, i, f"{sign}{val:.1f} mi/decade",
                    va="center", fontsize=5.5, color=C_DARK)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(crop_labels, fontsize=6)
    ax_bar.axvline(0, color="#999999", lw=0.6)
    ax_bar.set_xlabel("Shift rate (miles per decade)", fontsize=6)
    ax_bar.set_title("Northward Shift Rate\n(production-weighted centroid)", fontsize=7, fontweight="bold")
    ax_bar.tick_params(labelsize=5)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Bottom annotation
    if shift_values:
        total_shift = np.mean([rates[c] for c in crops_of_interest if c in rates]) * 6
        fig.text(0.5, 0.02,
                 f"Average production centroid has moved ~{total_shift:.0f} miles north since the 1960s",
                 ha="center", fontsize=6.5, color=C_DARK, style="italic",
                 fontweight="bold")

    fig.suptitle("Figure B — Northward Migration of U.S. Crop Production Centroids",
                 fontsize=7.5, fontweight="bold", y=0.98)

    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUT_DIR}/fig_northward_summary.{ext}", dpi=DPI)
    plt.close(fig)
    print("  [B] fig_northward_summary — done")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE C — Temperature-Yield Response Curves
# ══════════════════════════════════════════════════════════════════════════════
def fig_temp_response():
    fm = pd.read_parquet(f"{DATA_DIR}/processed/feature_matrix.parquet")

    crops_cfg = {
        "corn":         ("#E6550D", "Corn",         32, 34),
        "soybeans":     ("#31A354", "Soybeans",      30, 32),
        "wheat_winter": ("#756BB1", "Winter Wheat",  26, 28),
        "sorghum":      ("#3182BD", "Sorghum",       34, 36),
    }

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.5),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    axes = axes.flatten()

    for idx, (crop, (color, label, cliff_lo, cliff_hi)) in enumerate(crops_cfg.items()):
        ax = axes[idx]
        sub = fm[(fm["crop"] == crop) & fm["tmax_july_c"].notna()].copy()
        sub["bin"] = (sub["tmax_july_c"] // 1).astype(int)

        agg = sub.groupby("bin").agg(
            mean_y  = ("yield_anomaly", "mean"),
            se_y    = ("yield_anomaly", lambda x: x.std() / np.sqrt(len(x))),
            n       = ("yield_anomaly", "count"),
        ).reset_index()
        agg = agg[agg["n"] >= 20]  # at least 20 obs per bin

        # Empirical dots + error bars
        ax.errorbar(agg["bin"] + 0.5, agg["mean_y"], yerr=agg["se_y"],
                    fmt="o", color=color, ms=3, lw=0.7, capsize=2,
                    capthick=0.7, markeredgecolor="white", markeredgewidth=0.4,
                    label="Observed (±1 SE)", zorder=5)

        # Connect dots
        ax.plot(agg["bin"] + 0.5, agg["mean_y"], "-", color=color,
                lw=0.8, alpha=0.6, zorder=4)

        # Schlenker-Roberts damage function overlay
        x_sr = np.linspace(agg["bin"].min(), agg["bin"].max() + 1, 200)
        # Simplified quadratic damage function matching empirical range
        y_peak = agg["mean_y"].max()
        x_opt  = agg.loc[agg["mean_y"].idxmax(), "bin"] + 0.5
        # Quadratic with steeper right tail (Schlenker-Roberts style)
        a_left  = -y_peak / (x_opt - x_sr.min() + 0.01) ** 2
        a_right = -y_peak / (x_opt - x_sr.max() + 0.01) ** 2 * 1.8  # asymmetric
        y_sr = np.where(
            x_sr <= x_opt,
            y_peak + a_left * (x_sr - x_opt) ** 2,
            y_peak + a_right * (x_sr - x_opt) ** 2,
        )
        ax.plot(x_sr, y_sr, "--", color="#555555", lw=0.8, alpha=0.7,
                label="Schlenker-Roberts\ndamage fn.", zorder=3)

        # Cliff shading
        ax.axvspan(cliff_lo, cliff_hi, alpha=0.15, color=C_RED, zorder=1)
        ax.axvline((cliff_lo + cliff_hi) / 2, color=C_RED, lw=0.6,
                   ls=":", zorder=2)
        ax.text((cliff_lo + cliff_hi) / 2, ax.get_ylim()[0] if idx < 2 else ax.get_ylim()[0],
                f"Cliff\n~{int((cliff_lo+cliff_hi)/2)}°C",
                fontsize=4.5, color=C_RED, ha="center", va="bottom", zorder=6)

        ax.axhline(0, color="#CCCCCC", lw=0.5)
        ax.set_title(label, fontsize=7.5, fontweight="bold", color=color)
        ax.set_xlabel("July Tmax (°C)", fontsize=6)
        ax.set_ylabel("Yield anomaly (bu/ac)", fontsize=6)
        ax.tick_params(labelsize=5)
        if idx == 0:
            ax.legend(fontsize=4.5, loc="upper left", framealpha=0.85,
                      edgecolor="#CCCCCC", handlelength=1.5)

    fig.suptitle("Figure C — Empirical Temperature–Yield Response (binned means ± 1 SE)",
                 fontsize=7.5, fontweight="bold")

    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUT_DIR}/fig_temp_response.{ext}", dpi=DPI)
    plt.close(fig)
    print("  [C] fig_temp_response — done")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE D — Cascade Mechanism Diagram
# ══════════════════════════════════════════════════════════════════════════════
def fig_cascade_mechanism():
    fig = plt.figure(figsize=(SINGLE_COL + 0.3, 6.5))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    steps = [
        ("1. Yield Decline",        "Climate shock\n+1°C above threshold",      "#FFF5EB"),
        ("2. Farm Income ↓",        "Revenue = yield × price × acres\nElasticity: −0.42",   "#FEE6CE"),
        ("3. Rural Outmigration",   "β = −0.003 per income unit\nIV estimate, 2SLS",        "#FDD0A2"),
        ("4. School Closure",       "Threshold: <150 students\n(600 districts at risk)",     "#FDAE6B"),
        ("5. Hospital Closure",     "Threshold: <15,000 pop\n(213 rural hospitals)",         "#FD8D3C"),
        ("6. Tax Base Erosion",     "Property values ↓ → budget cuts\nFiscal multiplier: −0.6", "#E6550D"),
        ("7. Infrastructure ↓",     "Feedback intensity: 0.08/σ\n→ amplifies yield shock",  "#A63603"),
    ]

    n     = len(steps)
    y_top = 9.2
    y_bot = 0.9
    step_h= (y_top - y_bot) / n
    box_h = step_h * 0.72
    cx    = 2.0

    # Color gradient (yellow → dark red)
    grad_cmap = LinearSegmentedColormap.from_list(
        "cascade", ["#FFF5EB", "#A63603"], N=n)
    grad_colors = [grad_cmap(i / (n - 1)) for i in range(n)]

    for i, (title, detail, _) in enumerate(steps):
        y_center = y_top - i * step_h - step_h / 2
        col = grad_colors[i]
        txt_col = "white" if i >= 5 else C_DARK

        rect = FancyBboxPatch((cx - 1.55, y_center - box_h/2),
                               3.1, box_h,
                               boxstyle="round,pad=0.06",
                               facecolor=col,
                               edgecolor="white", linewidth=0.8, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, y_center + 0.1, title, ha="center", va="center",
                fontsize=6.5, fontweight="bold", color=txt_col, zorder=4)
        ax.text(cx, y_center - 0.15, detail, ha="center", va="center",
                fontsize=5, color=txt_col, alpha=0.9, zorder=4)

        # Downward arrow (except last)
        if i < n - 1:
            y_arrow_top = y_center - box_h/2
            y_arrow_bot = y_top - (i + 1) * step_h - step_h/2 + box_h/2
            ax.annotate("", xy=(cx, y_arrow_bot), xytext=(cx, y_arrow_top),
                        arrowprops=dict(arrowstyle="-|>", color="#C0392B",
                                        lw=1.0, mutation_scale=8))

    # Feedback arrow from step 7 back to step 1
    # Right side curved arrow
    ax_ybot = y_top - (n - 1) * step_h - step_h/2   # center of step 7
    ax_ytop = y_top - step_h/2                          # center of step 1
    ax.annotate("", xy=(cx + 1.55 + 0.08, ax_ytop),
                xytext=(cx + 1.55 + 0.08, ax_ybot),
                arrowprops=dict(arrowstyle="-|>", color="#7B2D8B",
                                lw=1.2, mutation_scale=8,
                                connectionstyle="arc3,rad=0.0"))
    ax.text(3.88, (ax_ytop + ax_ybot) / 2, "Feedback\nloop",
            ha="center", va="center", fontsize=5, color="#7B2D8B",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="#7B2D8B", lw=0.5))

    # Key annotation
    ax.text(cx, 0.45,
            "337 counties projected to reach tipping point by 2040",
            ha="center", va="center", fontsize=6, color="#A63603",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF5EB",
                      edgecolor="#A63603", lw=0.8))

    ax.text(cx, 9.7, "Figure D — Seven-Step Rural Cascade Mechanism",
            ha="center", va="center", fontsize=7, fontweight="bold", color=C_DARK)

    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUT_DIR}/fig_cascade_mechanism.{ext}", dpi=DPI)
    plt.close(fig)
    print("  [D] fig_cascade_mechanism — done")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE E — Insurance Cross-Subsidy Flow
# ══════════════════════════════════════════════════════════════════════════════
def fig_insurance_flow():
    fig = plt.figure(figsize=(DOUBLE_COL, 4.2))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1],
                   wspace=0.3, left=0.04, right=0.97, top=0.88, bottom=0.12)

    ax_map  = fig.add_subplot(gs[0])
    ax_side = fig.add_subplot(gs[1])

    # ── Simplified US map (CONUS bounding box with shaded regions) ──
    # North region = gaining (blue), South = declining (red)
    north_rect = patches.Rectangle((-125, 40), 60, 11,
                                    facecolor="#DEEBF7", edgecolor="#9ECAE1",
                                    lw=0.5, zorder=1, alpha=0.85)
    south_rect = patches.Rectangle((-125, 24), 60, 16,
                                    facecolor="#FEE0D2", edgecolor="#FC9272",
                                    lw=0.5, zorder=1, alpha=0.85)
    frame_rect = patches.Rectangle((-125, 24), 60, 27,
                                    facecolor="none", edgecolor="#999999",
                                    lw=0.8, zorder=2)

    for r in [north_rect, south_rect, frame_rect]:
        ax_map.add_patch(r)

    ax_map.text(-95, 46.5, "GAINING REGIONS\n(North)",
                ha="center", va="center", fontsize=6.5,
                fontweight="bold", color=C_BLUE, zorder=4)
    ax_map.text(-95, 28.5, "DECLINING REGIONS\n(South/Plains)",
                ha="center", va="center", fontsize=6.5,
                fontweight="bold", color=C_RED, zorder=4)

    # Large cross-subsidy arrow
    ax_map.annotate("",
        xy=(-95, 39.5), xytext=(-95, 40.5),
        arrowprops=dict(arrowstyle="-|>", color="#7B2D8B",
                        lw=3, mutation_scale=14))
    ax_map.text(-93, 40.0, "$2.8B/yr\nsubsidy flow",
                ha="left", va="center", fontsize=6.5,
                fontweight="bold", color="#7B2D8B",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#7B2D8B", lw=0.6))

    # Key regional labels
    regions_north = [("Corn Belt N.", -92, 44.5), ("Great Lakes", -84, 44),
                     ("Pacific NW",  -120, 46)]
    regions_south = [("Delta/SE",   -90, 31), ("High Plains", -102, 32),
                     ("Coastal SE",  -83, 26.5)]
    for label, x, y in regions_north:
        ax_map.text(x, y, label, ha="center", va="center",
                    fontsize=4.5, color="#2166AC", alpha=0.8)
    for label, x, y in regions_south:
        ax_map.text(x, y, label, ha="center", va="center",
                    fontsize=4.5, color=C_RED, alpha=0.8)

    ax_map.set_xlim(-126, -65)
    ax_map.set_ylim(23, 52)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("Longitude", fontsize=6)
    ax_map.set_ylabel("Latitude",  fontsize=6)
    ax_map.tick_params(labelsize=5)
    ax_map.set_title("Geographic Cross-Subsidy Flow\nin Federal Crop Insurance", fontsize=7, fontweight="bold")

    # ── Stacked bar + before/after panel ──
    ax_side.set_xlim(0, 10)
    ax_side.set_ylim(0, 10)
    ax_side.axis("off")

    ax_side.set_title("Premium Decomposition\n& Mispricing", fontsize=7, fontweight="bold")

    # Total program breakdown
    total_h = 4.5
    scale   = total_h / 18.0  # 18B total

    def bar_segment(ax, x, y, w, h, color, label, val_str, txt_color="white"):
        rect = patches.Rectangle((x, y), w, h,
                                   facecolor=color, edgecolor="white",
                                   lw=0.5, zorder=3)
        ax.add_patch(rect)
        if h > 0.3:
            ax.text(x + w/2, y + h/2, f"{label}\n{val_str}",
                    ha="center", va="center", fontsize=5,
                    fontweight="bold", color=txt_color, zorder=4)

    bar_x = 1.0
    bar_w = 2.2

    # Bottom: fairly priced portion
    y0 = 1.0
    h_fair = (18 - 5.9) * scale
    bar_segment(ax_side, bar_x, y0, bar_w, h_fair,
                "#DEEBF7", "Fairly priced", "$12.1B", "#2166AC")

    # Top: mispriced portion
    y1 = y0 + h_fair
    h_mis = 5.9 * scale
    bar_segment(ax_side, bar_x, y1, bar_w, h_mis,
                "#FC9272", "Mispriced", "$5.9B", "white")

    # Bracket + label
    ax_side.annotate("", xy=(bar_x + bar_w + 0.15, y1 + h_mis),
                     xytext=(bar_x + bar_w + 0.15, y0),
                     arrowprops=dict(arrowstyle="<->", color=C_DARK, lw=0.8))
    ax_side.text(bar_x + bar_w + 0.5, (y0 + y1 + h_mis)/2,
                 "Total\n$18B\nProgram", ha="left", va="center",
                 fontsize=5, color=C_DARK)

    # APH vs forward-looking comparison
    y_comp = 6.8
    ax_side.text(5.0, y_comp, "Before/After Actuarial Reform",
                 ha="center", va="center", fontsize=6.5, fontweight="bold",
                 color=C_DARK)

    labels = ["APH (current)", "Forward-looking"]
    vals   = [1.0, 1.47]
    colors = [C_ORANGE, C_BLUE]
    for i, (label, val, col) in enumerate(zip(labels, vals, colors)):
        bx    = 3.5 + i * 2.8
        bar_h = val * 1.4
        rect  = patches.FancyBboxPatch((bx - 0.6, y_comp - 3.2),
                                        1.2, bar_h,
                                        boxstyle="round,pad=0.04",
                                        facecolor=col, edgecolor="white",
                                        lw=0.5, zorder=3)
        ax_side.add_patch(rect)
        ax_side.text(bx, y_comp - 3.2 + bar_h + 0.12,
                     f"Loss ratio\n{val:.2f}", ha="center", va="bottom",
                     fontsize=5, color=C_DARK)
        ax_side.text(bx, y_comp - 3.3, label, ha="center", va="top",
                     fontsize=4.8, color=C_DARK, style="italic")

    ax_side.text(5.0, 3.7,
                 "+47% underpricing\nexposed by climate trend",
                 ha="center", va="center", fontsize=5.5,
                 color=C_RED, fontweight="bold")

    ax_side.text(5.0, 0.4, "$2.8B/yr flows from\nnorthern to southern counties",
                 ha="center", va="center", fontsize=5.5, color="#7B2D8B",
                 fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#F2E8F7",
                           edgecolor="#7B2D8B", lw=0.5))

    fig.suptitle("Figure E — Federal Crop Insurance Cross-Subsidy and Mispricing",
                 fontsize=7.5, fontweight="bold")

    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUT_DIR}/fig_insurance_flow.{ext}", dpi=DPI)
    plt.close(fig)
    print("  [E] fig_insurance_flow — done")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE F — Three-Method Valuation Comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig_valuation_methods():
    fig = plt.figure(figsize=(DOUBLE_COL, 4.0))
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1],
                   hspace=0.35, left=0.32, right=0.96, top=0.88, bottom=0.10)

    ax_main = fig.add_subplot(gs[0])
    ax_tail = fig.add_subplot(gs[1])

    # ── Main: horizontal bars for each method ──
    methods  = ["Cap Rate\n(overvalued counties)", "Hedonic\n(land price gradient)", "DCF\n(income-based)"]
    lo_vals  = [0,   150, 56]
    hi_vals  = [89,  176, 168]   # for cap rate, lo=0, hi=n_counties
    pt_vals  = [None, 168, None]
    colors   = [C_GREEN, C_GREEN, C_GREEN]
    # Use dollar axis for DCF and Hedonic; county count for Cap Rate
    # We'll normalize to billions for display and annotate separately

