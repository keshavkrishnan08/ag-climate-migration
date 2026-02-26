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

