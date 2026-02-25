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
