"""Summary figures from experiment JSONs.

Writes PDFs to results/figures_revision/ (insurance waterfall, migration forest, northern acreage).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.size": 2, "ytick.major.size": 2,
})

OUT = Path("results/figures_revision")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Insurance flow waterfall (replaces table 23)
# ============================================================
flow_labels = ["Gross gap\n(frozen APH)", "Rolling APH\nabsorbs", "TAY absorbs", "SCO + ECO", "Residual\n(reform-elim.)", "Cross-subsidy\n(× 62% federal)"]
flow_values = [6.6, -2.0, -0.9, 0.01, 3.7, 1.6]
flow_colors = ["#1f77b4", "#999", "#999", "#999", "#2ca02c", "#d62728"]

fig, ax = plt.subplots(figsize=(4.5, 2.4))
cum = 0
for i, (label, val, color) in enumerate(zip(flow_labels, flow_values, flow_colors)):
    if i == 0:
        ax.bar(i, val, color=color, edgecolor="black", linewidth=0.5)
        ax.text(i, val + 0.15, f"${val}B", ha="center", fontsize=7)
        cum = val
    elif i == 4:  # Residual, final
        ax.bar(i, val, color=color, edgecolor="black", linewidth=0.7)
        ax.text(i, val + 0.15, f"${val}B", ha="center", fontsize=7, fontweight="bold")
    elif i == 5:  # Cross-subsidy
        ax.bar(i, val, color=color, edgecolor="black", linewidth=0.7)
        ax.text(i, val + 0.15, f"${val}B", ha="center", fontsize=7, fontweight="bold")
    else:  # absorbed components shown going down
        new_cum = cum + val
        ax.bar(i, val, bottom=cum, color=color, edgecolor="black", linewidth=0.5)
        ax.text(i, cum + val/2, f"${val:+.2f}B" if i == 3 else f"${val:+.1f}B",
                ha="center", va="center", fontsize=7)
        cum = new_cum

ax.set_xticks(range(len(flow_labels)))
ax.set_xticklabels(flow_labels, fontsize=6.5)
ax.set_ylabel("Annual flow (2023 USD, B)", fontsize=8)
ax.set_title("Insurance mispricing decomposition (2040–2050, SSP2-4.5)", fontsize=8)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylim(-1, 7.5)
plt.tight_layout()
plt.savefig(OUT / "fig_insurance_waterfall.pdf", bbox_inches="tight")
plt.close()
print("Insurance waterfall saved.")

# ============================================================
# Migration IV forest plot (replaces table 16)
# ============================================================
specs = [
    ("3-yr horizon",        0.024, 0.011, 78),
    ("5-yr (headline)",     0.049, 0.015, 60),
    ("5-yr non-overlap",    0.059, 0.022, None),
    ("5-yr WCB ($B=9999$)", 0.049, 0.013, None),
    ("High-intensity tercile (3-yr)", 0.057, 0.027, None),
    ("Pre-2020 window",     0.020, 0.010, None),
]
fig, ax = plt.subplots(figsize=(4.5, 2.4))
y_pos = np.arange(len(specs))
for i, (name, b, se, F) in enumerate(specs):
    if se is not None:
        ax.errorbar(b, i, xerr=1.96 * se, fmt="o", color="#1f77b4",
                    markersize=5, capsize=2, linewidth=0.8)
    else:
        ax.plot(b, i, "o", color="#1f77b4", markersize=5)
    ax.text(b + 1.96 * (se or 0.005) + 0.005, i, f"$\\hat\\beta={b:.3f}$",
            va="center", fontsize=6.5)
ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_yticks(y_pos)
ax.set_yticklabels([s[0] for s in specs], fontsize=7)
ax.set_xlabel("Migration IV coefficient $\\hat\\beta$", fontsize=8)
ax.set_title("Migration IV across specifications, 444 farming-dependent counties", fontsize=8)
ax.set_xlim(-0.005, 0.13)
plt.tight_layout()
plt.savefig(OUT / "fig_migration_forest.pdf", bbox_inches="tight")
plt.close()
print("Migration forest saved.")

# ============================================================
# Northern acreage 1980-2005 (replaces table 22)
# ============================================================
import pandas as pd
df = pd.read_parquet("data/raw/nass/nass_county_yields.parquet",
                     columns=["fips", "year", "crop", "acres_harvested"])
df = df.dropna(subset=["fips", "year", "acres_harvested"])
df["fips"] = df["fips"].astype(str).str.zfill(5)
df["state_fips"] = df["fips"].str[:2]
df["crop_upper"] = df["crop"].str.upper().str.strip()
NORTHERN = ["27", "55", "38", "46", "30", "16"]
CROPS = ["CORN", "SOYBEANS", "SPRING WHEAT"]
sub = df[(df["state_fips"].isin(NORTHERN)) & (df["crop_upper"].isin(CROPS))]
sub = sub[(sub["year"] >= 1980) & (sub["year"] <= 2005)]
annual = sub.groupby("year")["acres_harvested"].sum().reset_index()

fig, ax = plt.subplots(figsize=(4.5, 2.0))
ax.plot(annual["year"], annual["acres_harvested"] / 1e6, "o-",
        color="#1f77b4", markersize=4, linewidth=1)
# Add log-linear fit
slope = np.polyfit(annual["year"].values,
                   np.log(annual["acres_harvested"].values), 1)[0]
growth_pct = (np.exp(slope) - 1) * 100
ax.set_xlabel("Year", fontsize=8)
ax.set_ylabel("Acres (millions)", fontsize=8)
ax.set_title(f"Northern corn + soybean + spring-wheat acreage 1980–2005 ({growth_pct:.1f}%/yr)",
             fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "fig_northern_acreage.pdf", bbox_inches="tight")
plt.close()
print("Northern acreage saved.")
print(f"\nAll SI graphics saved to {OUT}")
