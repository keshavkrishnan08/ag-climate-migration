"""Phase 6: All 12 publication figures.

All figures at 300 DPI. Nature Food formatting:
    Single column = 88mm, double column = 180mm
    All text in figures: Arial 7pt minimum.

Figure specifications from PRD Section 8.
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
import seaborn as sns
from loguru import logger
import yaml

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    logger.warning("geopandas not available — choropleth figures will use scatter fallback")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
PROJECTIONS_DIR = PROJECT_ROOT / 'data' / 'projections'
RESULTS_DIR = PROJECT_ROOT / 'results'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

# Nature Food style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Column widths in inches (from mm)
SINGLE_COL = 88 / 25.4   # ~3.46 inches
DOUBLE_COL = 180 / 25.4   # ~7.09 inches


def save_figure(fig, name: str, output_dir: Path = None):
    """Save figure as both PDF and PNG at 300 DPI.

    Args:
        fig: matplotlib Figure object.
        name: Figure name (without extension).
        output_dir: Output directory.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / f'{name}.pdf', format='pdf')
    fig.savefig(output_dir / f'{name}.png', format='png')
    plt.close(fig)
    logger.info(f"Saved figure: {name} → {output_dir}")


# ---------------------------------------------------------------------------
# Shared choropleth utilities
# ---------------------------------------------------------------------------

# Path to county shapefile (downloaded from Census TIGER/Line)
