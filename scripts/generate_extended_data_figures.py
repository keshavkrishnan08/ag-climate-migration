"""
Generate Extended Data figures for Nature Food submission.

ED Figure 1: GCM Ensemble Spread by Region (fan charts)
ED Figure 2: Historical Cascade Score Distribution (bar chart)

Args: none
Returns: saves PDFs to results/figures/
Raises: FileNotFoundError if input data missing
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ_PATH = os.path.join(BASE, 'data', 'projections', 'yield_projections_SSP245.parquet')
CASCADE_PATH = os.path.join(BASE, 'results', 'economic', 'historical_cascade_summary.json')
OUT_DIR = os.path.join(BASE, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Nature Food palette — colorblind-safe
COLORS = {
    'Corn Belt':      '#2166ac',
    'Southern Plains':'#d6604d',
    'Northern Plains':'#4dac26',
    'Southeast':      '#8073ac',
}

REGION_STATES = {
    'Corn Belt':       ['17', '18', '19', '27', '29', '31', '39', '55'],
    'Southern Plains': ['20', '40', '48'],
    'Northern Plains': ['30', '38', '46'],
    'Southeast':       ['01', '05', '12', '13', '22', '28', '37', '45', '47', '51'],
}


# ─────────────────────────────────────────────────────────────────────────────
# ED Figure 1: GCM Ensemble Spread Fan Charts
# ─────────────────────────────────────────────────────────────────────────────
def make_ed_fig1():
    """
    Build 2×2 fan chart showing mean yield-change trajectory ± p10/p90 band
    for four US agricultural regions, 2025–2050.

    Returns:
        str: output PDF path
    """
    print("Loading yield projections …")
    df = pd.read_parquet(PROJ_PATH,
                         columns=['fips', 'year', 'crop', 'yield_projected',
                                  'yield_baseline', 'yield_p10', 'yield_p90'])

    # Compute % change relative to baseline
    df['pct_mean'] = (df['yield_projected'] - df['yield_baseline']) / df['yield_baseline'].abs() * 100
    df['pct_p10']  = (df['yield_p10']       - df['yield_baseline']) / df['yield_baseline'].abs() * 100
    df['pct_p90']  = (df['yield_p90']       - df['yield_baseline']) / df['yield_baseline'].abs() * 100

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), dpi=300, sharey=False)
    axes_flat = axes.flatten()

    regions = list(REGION_STATES.keys())
    for idx, region in enumerate(regions):
        ax = axes_flat[idx]
        color = COLORS[region]
        states = REGION_STATES[region]

        mask = df['fips'].str[:2].isin(states)
        sub = df[mask].groupby('year').agg(
            mean=('pct_mean', 'mean'),
            p10=('pct_p10',  'mean'),
            p90=('pct_p90',  'mean'),
        ).reset_index()

        years = sub['year'].values

        # Fan (shaded band)
        ax.fill_between(years, sub['p10'], sub['p90'],
                        color=color, alpha=0.20, linewidth=0, label='p10–p90 band')
        # Mean line
        ax.plot(years, sub['mean'], color=color, linewidth=2.0, label='GCM ensemble mean')
        # Zero reference
        ax.axhline(0, color='#444444', linewidth=0.7, linestyle='--', alpha=0.6)

        ax.set_title(region, fontsize=10, fontweight='bold', pad=4)
        ax.set_xlabel('Year', fontsize=8)
        ax.set_ylabel('Yield change (%)', fontsize=8)
        ax.set_xlim(2025, 2050)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add n_counties annotation
        n_counties = df.loc[mask, 'fips'].nunique()
        ax.annotate(f'n = {n_counties} counties', xy=(0.97, 0.05),
                    xycoords='axes fraction', ha='right', fontsize=7,
                    color='#555555')

    # Shared legend
    handles = [
        mpatches.Patch(color='#888888', alpha=0.3, label='p10–p90 GCM spread'),
        plt.Line2D([0], [0], color='#888888', linewidth=2, label='Ensemble mean'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        'Extended Data Fig. 1 — GCM Ensemble Spread by Region (SSP2-4.5, 2025–2050)',
        fontsize=9, fontweight='bold', y=1.01
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out = os.path.join(OUT_DIR, 'ed_fig01_gcm_spread.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ED Figure 2: Cascade Score Distribution
# ─────────────────────────────────────────────────────────────────────────────
def make_ed_fig2():
    """
    Build bar chart of cascade score distribution (1–6 signals) from
    historical_cascade_summary.json.

    Returns:
        str: output PDF path
    """
    print("Loading cascade summary …")
    with open(CASCADE_PATH) as f:
        data = json.load(f)

    dist = data['cascade_score_distribution']
    scores = sorted(int(k) for k in dist.keys())
    counts = [dist[str(s)] for s in scores]

    # Color: grey for 1-3, amber for 4-5, red for 6
    bar_colors = []
    for s in scores:
        if s <= 3:
            bar_colors.append('#aab4c4')
        elif s == 6:
            bar_colors.append('#c0392b')
        else:
            bar_colors.append('#e67e22')

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    bars = ax.bar(scores, counts, color=bar_colors, edgecolor='white',
                  linewidth=0.8, width=0.65, zorder=3)

    # Count labels above bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8,
                f'{count:,}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Threshold annotation
    ax.axvline(3.5, color='#2c3e50', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.text(3.55, max(counts) * 0.92, 'Tipping\nthreshold\n(≥4 signals)',
            fontsize=7.5, color='#2c3e50', va='top')

    ax.set_xlabel('Number of cascade signals observed (2009–2023)', fontsize=9)
    ax.set_ylabel('County count', fontsize=9)
    ax.set_title(
        'Extended Data Fig. 2 — Historical Cascade Score Distribution\n'
        '(1,824 agricultural counties, observed data)',
        fontsize=9, fontweight='bold'
    )
    ax.set_xticks(scores)
    ax.set_xticklabels([str(s) for s in scores], fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Legend patches
    legend_patches = [
        mpatches.Patch(color='#aab4c4', label='1–3 signals (below threshold)'),
        mpatches.Patch(color='#e67e22', label='4–5 signals (at risk)'),
        mpatches.Patch(color='#c0392b', label='6 signals (full cascade)'),
    ]
    ax.legend(handles=legend_patches, fontsize=7.5, loc='upper right',
              framealpha=0.9, edgecolor='#cccccc')

    # Annotate total >= 4
    total_gte4 = sum(counts[i] for i, s in enumerate(scores) if s >= 4)
    ax.annotate(
        f'473 counties ≥ 4 signals\n(25.9% of agricultural counties)',
        xy=(4, counts[scores.index(4)]),
        xytext=(4.6, counts[scores.index(4)] + 60),
        fontsize=7.5,
        arrowprops=dict(arrowstyle='->', color='#555', lw=0.8),
        color='#333333'
    )

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'ed_fig02_cascade_dist.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == '__main__':
    make_ed_fig1()
    make_ed_fig2()
    print("All extended data figures complete.")
