"""Robustness gallery figures from adversarial experiment JSONs.

Reads results/revision/adversarial/*.json; writes PDFs to results/figures_revision/.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

OUT = Path("results/figures_revision")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Figure E55: SEM loadings across 4 nested partial-out depths
# ============================================================
e55 = json.load(open("results/revision/adversarial/e55_sem_partialouts.json"))
depths = ["P0\nbaseline", "P1\n−July-Tmax", "P2\n+IV shock\n+FPR FE", "P3\n−literal\nyield input"]
loadings = np.array([
    e55["baseline_no_partial"]["loadings"],
    e55["P1_julyTmax_partialed"]["loadings"],
    e55["P2_julyTmax_plus_shiftshare_plus_FPR_FE"]["loadings"],
    e55["P3_literal_yield_projection_partialed"]["loadings"],
])
channels = ["Stranded\nvalue", "Insurance\nmispricing", "Rural\ndecline", "Northern\nopportunity"]

fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(len(depths))
width = 0.2
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for i, ch in enumerate(channels):
    ax.bar(x + (i - 1.5) * width, loadings[:, i], width, label=ch, color=colors[i])
ax.axhline(0.25, color="black", linestyle="--", linewidth=0.7,
           label="0.25 substantive-loadings cutoff")
ax.set_xticks(x)
ax.set_xticklabels(depths, fontsize=8)
ax.set_ylabel("Single-factor standardised loading")
ax.set_title("E55. SEM single-factor loadings survive three nested partial-outs", fontsize=10)
ax.set_ylim(0, 0.8)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=7, frameon=False)
plt.tight_layout()
plt.savefig(OUT / "fig_e55_sem_partialouts.pdf", bbox_inches="tight")
plt.close()
print("E55 fig saved.")

# ============================================================
# Figure E56: DCF and hedonic bootstrap CI overlap
# ============================================================
e56 = json.load(open("results/revision/adversarial/e56_dcf_hedonic_overlap.json"))
specs = [
    ("DCF conservative\n($r=4\\%$, no floor)", e56["DCF_conservative"]["central"],
     e56["DCF_conservative"]["ci_full_95"]),
    ("DCF central\n($r=5\\%$, floor)", e56["DCF_central"]["central"],
     e56["DCF_central"]["ci_full_95"]),
    ("Hedonic\n(soil + irrigation)", e56["Hedonic_soil_irrigation"]["central"],
     e56["Hedonic_soil_irrigation"]["ci_95"]),
]
fig, ax = plt.subplots(figsize=(7, 3.0))
y_pos = np.arange(len(specs))
for i, (name, central, ci) in enumerate(specs):
    color = "#1f77b4" if "DCF" in name else "#d62728"
    ax.plot(ci, [i, i], "-", linewidth=2, color=color)
    ax.plot([ci[0], ci[1]], [i, i], "|", markersize=12, color=color)
    ax.plot(central, i, "o", markersize=8, color=color)
    ax.text(central, i + 0.18, f"${central:.1f}B", ha="center", va="bottom",
            fontsize=8, color=color)
overlap = e56["overlap_region"]
ax.axvspan(overlap[0], overlap[1], alpha=0.15, color="green",
           label=f"Overlap [${overlap[0]:.1f}, ${overlap[1]:.1f}]B")
ax.set_yticks(y_pos)
ax.set_yticklabels([s[0] for s in specs], fontsize=8)
ax.set_xlabel("Stranded farmland value, billion 2023 USD")
ax.set_title("E56. DCF and hedonic 95\\% bootstrap intervals overlap", fontsize=10)
ax.set_xlim(30, 115)
ax.legend(loc="lower right", fontsize=7, frameon=False)
plt.tight_layout()
plt.savefig(OUT / "fig_e56_dcf_hedonic_overlap.pdf", bbox_inches="tight")
plt.close()
print("E56 fig saved.")

# ============================================================
# Figure E58: First-stage F by year
# ============================================================
e58 = json.load(open("results/revision/adversarial/e58_iv_loo_year.json"))
F_by_year = e58["first_stage_F_by_year"]
years = sorted([int(y) for y in F_by_year.keys()])
F_vals = [F_by_year[str(y)] for y in years]
stock_yogo = e58["stock_yogo_cutoff_F"]

fig, ax = plt.subplots(figsize=(7, 3.0))
bars = ax.bar(years, F_vals, color="#1f77b4", edgecolor="black", linewidth=0.5)
# Highlight 2012 in red
for i, y in enumerate(years):
    if y == 2012:
        bars[i].set_color("#d62728")
ax.axhline(stock_yogo, color="black", linestyle="--", linewidth=0.8,
           label=f"Stock-Yogo $F = {stock_yogo}$ cutoff")
ax.set_xlabel("Year")
ax.set_ylabel("First-stage $F$")
ax.set_title("E58. Migration IV first-stage $F$ above Stock-Yogo cutoff in 15/15 years",
             fontsize=10)
ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=8)
ax.legend(loc="upper right", fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(OUT / "fig_e58_iv_first_stage_F.pdf", bbox_inches="tight")
plt.close()
print("E58 fig saved.")

# ============================================================
# Figure E61: Yield R^2 vs literature
# ============================================================
lit_data = [
    ("Roberts-Schlenker\n(2013)", 0.18),
    ("Burke-Emerick\n(2016)", 0.21),
    ("Lobell et al.\n(2014)", 0.32),
    ("Ortiz-Bobea\net al.\n(2021)", 0.41),
    ("This paper\n(climate response)", 0.41),
    ("Schlenker-Roberts\n(2009, levels)", 0.71),
    ("This paper\n(levels, median)", 0.68),
]
names = [d[0] for d in lit_data]
vals = [d[1] for d in lit_data]
colors = ["#888"] * 4 + ["#d62728"] + ["#888"] + ["#d62728"]

fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(len(names))
ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=7.5)
ax.set_ylabel("Held-out $R^2$ (climate-response or levels)")
ax.set_title("E61. Yield-model $R^2$ vs literature", fontsize=10)
ax.axhline(0.5, color="green", linestyle=":", linewidth=0.7,
           label="0.5 AgMIP county-scale benchmark")
ax.set_ylim(0, 0.85)
ax.legend(loc="upper left", fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(OUT / "fig_e61_yield_R2_literature.pdf", bbox_inches="tight")
plt.close()
print("E61 fig saved.")

print("\nAll adversarial gallery figures saved to results/figures_revision/")
