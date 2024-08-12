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
_COUNTY_SHP = DATA_RAW / 'census' / 'us_counties_20m.shp'
# Excluded state FIPS (non-CONUS)
_NON_CONUS_STATES = {'02', '15', '72', '78', '66', '60', '69'}


def _load_conus_counties() -> 'gpd.GeoDataFrame':
    """Load CONUS county shapefile from Census TIGER/Line.

    Returns:
        GeoDataFrame with CONUS counties; column 'fips' = 5-digit GEOID.

    Raises:
        FileNotFoundError: if shapefile is missing and geopandas unavailable.
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas required for choropleth maps")
    counties = gpd.read_file(_COUNTY_SHP)
    counties = counties.rename(columns={'GEOID': 'fips'})
    counties = counties[~counties['STATEFP'].isin(_NON_CONUS_STATES)].copy()
    counties['fips'] = counties['fips'].astype(str).str.zfill(5)
    return counties


def _choropleth(ax, counties_geo: 'gpd.GeoDataFrame', col: str,
                cmap: str, vmin: float, vmax: float,
                title: str, unit: str,
                missing_color: str = '#cccccc') -> None:
    """Draw a county-level choropleth on *ax*.

    Args:
        ax: Matplotlib axes.
        counties_geo: GeoDataFrame with 'geometry' and *col* pre-joined.
        col: Column to color-map.
        cmap: Matplotlib colormap name.
        vmin: Colormap minimum value.
        vmax: Colormap maximum value.
        title: Axes title.
        unit: Unit string for colorbar label.
        missing_color: Fill for counties without data.
    """
    # Counties without data
    mask_missing = counties_geo[col].isna()
    counties_geo[mask_missing].plot(ax=ax, color=missing_color,
                                    linewidth=0.05, edgecolor='white')
    # Counties with data
    counties_geo[~mask_missing].plot(
        ax=ax, column=col, cmap=cmap, vmin=vmin, vmax=vmax,
        linewidth=0.05, edgecolor='white', legend=False
    )
    sm = cm.ScalarMappable(cmap=cmap,
                           norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.7)
    cb.set_label(unit, fontsize=6)
    cb.ax.tick_params(labelsize=5)
    ax.set_title(title, fontweight='bold', fontsize=8)
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Figure 1: The Northward Migration Is Already Underway
# ---------------------------------------------------------------------------
def _load_county_centroids() -> pd.DataFrame:
    """Load county centroids from Census Gazetteer file.

    Returns:
        DataFrame with columns: fips (str 5-digit), lat (float), lon (float).
    """
    gaz_path = DATA_RAW / 'census' / '2023_Gaz_counties_national.txt'
    gaz = pd.read_csv(gaz_path, sep='\t', dtype=str)
    gaz.columns = gaz.columns.str.strip()
    gaz = gaz.rename(columns={'GEOID': 'fips', 'INTPTLAT': 'lat', 'INTPTLONG': 'lon'})
    gaz['fips'] = gaz['fips'].str.zfill(5)
    gaz['lat'] = pd.to_numeric(gaz['lat'].str.strip(), errors='coerce')
    gaz['lon'] = pd.to_numeric(gaz['lon'].str.strip(), errors='coerce')
    # CONUS only: exclude AK (02), HI (15), PR (72) and territories
    excluded = {'02', '15', '72', '78', '66', '60', '69'}
    gaz = gaz[~gaz['fips'].str[:2].isin(excluded)]
    logger.info(f"Loaded {len(gaz)} CONUS county centroids from Gazetteer")
    return gaz[['fips', 'lat', 'lon']].copy()


def _compute_production_centroids(nass: pd.DataFrame,
                                  centroids: pd.DataFrame) -> pd.DataFrame:
    """Compute production-weighted centroid latitude per crop-year.

    For each crop and year, the centroid latitude is:
        sum(county_lat * county_acres) / sum(county_acres)

    Args:
        nass: Deduplicated NASS yields with fips, year, crop, acres_harvested.
        centroids: County centroids with fips, lat.

    Returns:
        DataFrame with year, crop, centroid_lat columns.
    """
    merged = nass.merge(centroids[['fips', 'lat']], on='fips', how='inner')
    merged = merged.dropna(subset=['acres_harvested', 'lat'])
    merged = merged[merged['acres_harvested'] > 0]

    merged['weighted_lat'] = merged['lat'] * merged['acres_harvested']
    agg = merged.groupby(['year', 'crop']).agg(
        total_weighted_lat=('weighted_lat', 'sum'),
        total_acres=('acres_harvested', 'sum')
    ).reset_index()
    agg['centroid_lat'] = agg['total_weighted_lat'] / agg['total_acres']
    return agg[['year', 'crop', 'centroid_lat']].copy()


def _compute_frontier_latitude(nass: pd.DataFrame,
                               centroids: pd.DataFrame,
                               percentile: float = 90) -> pd.DataFrame:
    """Compute the northern frontier latitude for each crop-year.

    The frontier is the latitude below which `percentile`% of total
    harvested acreage lies. A northward shift in this line means
    production is expanding into higher latitudes.

    Args:
        nass: Deduplicated NASS yields with fips, year, crop, acres_harvested.
        centroids: County centroids with fips, lat.
        percentile: Cumulative production percentile (default 90).

    Returns:
        DataFrame with year, crop, frontier_lat columns.
    """
    merged = nass.merge(centroids[['fips', 'lat']], on='fips', how='inner')
    merged = merged.dropna(subset=['acres_harvested', 'lat'])
    merged = merged[merged['acres_harvested'] > 0]

    records = []
    for (yr, crop), grp in merged.groupby(['year', 'crop']):
        sorted_grp = grp.sort_values('lat')
        cum_acres = sorted_grp['acres_harvested'].cumsum()
        total = cum_acres.iloc[-1]
        threshold = total * (percentile / 100.0)
        idx = (cum_acres >= threshold).idxmax()
        records.append({
            'year': yr,
            'crop': crop,
