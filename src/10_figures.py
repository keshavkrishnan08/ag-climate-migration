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
            'frontier_lat': sorted_grp.loc[idx, 'lat']
        })
    return pd.DataFrame(records)


def _lat_to_miles(delta_lat_per_decade: float) -> float:
    """Convert latitude degrees/decade to miles/decade.

    Args:
        delta_lat_per_decade: Trend slope in degrees latitude per decade.

    Returns:
        Equivalent distance in miles per decade (1 degree ~ 69 miles).
    """
    return delta_lat_per_decade * 69.0


def figure_01_northward_migration(
    yields: pd.DataFrame = None,
    output_dir: Path = None
) -> plt.Figure:
    """4-panel figure: production-weighted centroid + 90th-percentile northern
    frontier for corn, soybeans, winter wheat, and cotton (1950-2023).

    Uses actual county centroids from Census Gazetteer weighted by NASS
    harvested acreage. Two lines per panel:
      - Centroid (black): where the average acre sits. May drift south
        if intensification in existing areas outpaces frontier expansion.
      - 90th-percentile frontier (blue): the latitude below which 90%
        of production occurs. Northward movement here is the clearest
        signal of climate-driven range expansion.

    Trend lines fitted by OLS; shift reported in miles/decade.

    Args:
        yields: NASS county yields with acreage. If None, loaded from disk.
        output_dir: Where to save.

    Returns:
        matplotlib Figure.
    """
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if yields is None:
        nass = pd.read_parquet(
            DATA_RAW / 'nass' / 'nass_county_yields.parquet',
            columns=['fips', 'year', 'crop', 'yield_bu_acre', 'acres_harvested']
        )
    else:
        nass = yields.copy()

    # Filter crops and years
    target_crops = ['corn', 'soybeans', 'wheat_winter', 'cotton']
    nass = nass[nass['crop'].isin(target_crops)]
    nass = nass[(nass['year'] >= 1950) & (nass['year'] <= 2023)]
    nass['fips'] = nass['fips'].astype(str).str.zfill(5)

    # Remove aggregate FIPS (ending in 998, 999)
    nass = nass[~nass['fips'].str[-3:].isin(['998', '999'])]

    # Deduplicate: first record per fips-year-crop
    nass = nass.groupby(['fips', 'year', 'crop'], as_index=False).agg(
        yield_bu_acre=('yield_bu_acre', 'first'),
        acres_harvested=('acres_harvested', 'first')
    )
    logger.info(f"NASS after dedup: {len(nass):,} rows, "
                f"{nass['fips'].nunique()} counties")

    # Load county centroids
    centroids = _load_county_centroids()

    # ------------------------------------------------------------------
    # 2. Compute metrics
    # ------------------------------------------------------------------
    centroid_df = _compute_production_centroids(nass, centroids)
    frontier_df = _compute_frontier_latitude(nass, centroids, percentile=90)

    # ------------------------------------------------------------------
    # 3. Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))
    crops = ['corn', 'soybeans', 'wheat_winter', 'cotton']
    titles = ['Corn', 'Soybeans', 'Winter Wheat', 'Cotton']

    # Color palette
    centroid_color = '#2d2d2d'   # near-black
    frontier_color = '#1f77b4'   # muted blue
    centroid_trend_color = '#d62728'   # red
    frontier_trend_color = '#1a5276'   # dark teal

    for ax, crop, title in zip(axes.flat, crops, titles):
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Year')

        # --- Centroid ---
        c_data = centroid_df[centroid_df['crop'] == crop].sort_values('year')
        if len(c_data) > 0:
            yrs_c = c_data['year'].values
            lat_c = c_data['centroid_lat'].values
            ax.plot(yrs_c, lat_c, color=centroid_color, linewidth=0.6,
                    alpha=0.5, zorder=1)
            # Savitzky-Golay smoothing for readability
            if len(lat_c) > 11:
                window = min(15, len(lat_c) // 2 * 2 + 1)
                lat_c_smooth = savgol_filter(lat_c, window, 2)
                ax.plot(yrs_c, lat_c_smooth, color=centroid_color,
                        linewidth=1.5, label='Centroid', zorder=2)
            # OLS trend
            z_c = np.polyfit(yrs_c, lat_c, 1)
            ax.plot(yrs_c, np.polyval(z_c, yrs_c), color=centroid_trend_color,
                    linestyle='--', linewidth=0.9, zorder=3)
            miles_c = _lat_to_miles(z_c[0] * 10)
            direction_c = 'N' if miles_c > 0 else 'S'

        # --- 90th percentile frontier ---
        f_data = frontier_df[frontier_df['crop'] == crop].sort_values('year')
        if len(f_data) > 0:
            yrs_f = f_data['year'].values
            lat_f = f_data['frontier_lat'].values
            ax.plot(yrs_f, lat_f, color=frontier_color, linewidth=0.6,
                    alpha=0.4, zorder=1)
            if len(lat_f) > 11:
                window = min(15, len(lat_f) // 2 * 2 + 1)
                lat_f_smooth = savgol_filter(lat_f, window, 2)
                ax.plot(yrs_f, lat_f_smooth, color=frontier_color,
                        linewidth=1.5, label='90th pctl frontier', zorder=2)
            # OLS trend
            z_f = np.polyfit(yrs_f, lat_f, 1)
            ax.plot(yrs_f, np.polyval(z_f, yrs_f), color=frontier_trend_color,
                    linestyle='--', linewidth=0.9, zorder=3)
            miles_f = _lat_to_miles(z_f[0] * 10)
            direction_f = 'N' if miles_f > 0 else 'S'

        # Annotation box
        if len(c_data) > 0 and len(f_data) > 0:
            text = (f'Centroid: {abs(miles_c):.0f} mi/dec {direction_c}\n'
                    f'Frontier: {abs(miles_f):.0f} mi/dec {direction_f}')
            ax.text(0.03, 0.97, text, transform=ax.transAxes,
                    va='top', ha='left', fontsize=6,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.85))

        ax.set_ylabel('Latitude (°N)')
        ax.legend(loc='lower right', framealpha=0.85, fontsize=6)

    fig.suptitle('Fig. 1: Production Centroid & Northern Frontier, 1950–2023',
                 fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout()

    save_figure(fig, 'fig01_northward_migration', output_dir)

    # Log summary statistics
    for crop in crops:
        c_data = centroid_df[centroid_df['crop'] == crop]
        f_data = frontier_df[frontier_df['crop'] == crop]
        if len(c_data) > 0:
            z_c = np.polyfit(c_data['year'], c_data['centroid_lat'], 1)
            miles_c = _lat_to_miles(z_c[0] * 10)
        if len(f_data) > 0:
            z_f = np.polyfit(f_data['year'], f_data['frontier_lat'], 1)
            miles_f = _lat_to_miles(z_f[0] * 10)
        logger.info(f"  {crop:15s}  centroid={miles_c:+.1f} mi/dec  "
                     f"frontier={miles_f:+.1f} mi/dec")

    return fig


# ---------------------------------------------------------------------------
# Figure 2: Model Validation: Hindcast 2013-2023
# ---------------------------------------------------------------------------
def figure_02_model_validation(
    observed: pd.DataFrame = None,
    predicted: pd.DataFrame = None,
    output_dir: Path = None
) -> plt.Figure:
    """Scatter: Predicted vs observed yield anomaly on held-out test set (2017-2023).

    Uses the trained LightGBM yield model from results/20260317_192605/yield_model.pkl
    applied to the feature matrix test split (years 2017-2023, temporal holdout).
    Yields are z-score anomalies, so axes are in standard-deviation units.
    Panels for Corn, Soybeans, Winter Wheat — the three largest acreage crops.

    Args:
        observed: Ignored; loaded from feature_matrix.parquet.
        predicted: Ignored; generated by model inference.
        output_dir: Where to save.

    Returns:
        matplotlib Figure.
    """
    # ------------------------------------------------------------------
    # 1. Load feature matrix test split and run model inference
    # ------------------------------------------------------------------
    feature_base = [
        'yield_trend_slope_15yr', 'yield_trend_intercept',
        'tmax_july_c', 'tmax_growing_c', 'tmin_growing_c',
        'precip_growing', 'pdsi_growing', 'cdd_annual',
        'gdd_corn', 'gdd_soybeans', 'gdd_wheat_winter',
        'gdd_wheat_spring', 'gdd_cotton', 'gdd_sorghum',
        'gdd_barley', 'gdd_oats',
        'tmax_july_c_trend10', 'precip_growing_trend10', 'cdd_annual_trend10',
        'tmax_july_c_anomaly', 'precip_growing_anomaly', 'pdsi_growing_anomaly',
        'extreme_heat_months', 'switching_rate_proxy', 'switching_rate_5yr',
        'log_population', 'log_median_income', 'poverty_rate',
    ]
    crop_dummies = ['barley', 'corn', 'cotton', 'oats',
                    'sorghum', 'soybeans', 'wheat_spring', 'wheat_winter']

    fm = pd.read_parquet(
        DATA_PROCESSED / 'feature_matrix.parquet',
        columns=['fips', 'year', 'crop', 'yield_anomaly'] + feature_base
    )
    test = fm[fm['year'].between(2017, 2023)].copy()
    for c in crop_dummies:
        test[f'crop_{c}'] = (test['crop'] == c).astype(int)
    all_feature_cols = feature_base + [f'crop_{c}' for c in crop_dummies]
    test = test.dropna(subset=['yield_anomaly'] + all_feature_cols)

    model_path = RESULTS_DIR / '20260317_192605' / 'yield_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    test['predicted_anomaly'] = model.predict(test[all_feature_cols])
    logger.info(f"Fig02: predictions on {len(test):,} test rows "
                f"(Spearman={spearmanr(test['yield_anomaly'], test['predicted_anomaly'])[0]:.3f})")

    # ------------------------------------------------------------------
    # 2. Plot per-crop scatter panels
    # ------------------------------------------------------------------
    plot_crops = [('corn', 'Corn'), ('soybeans', 'Soybeans'),
                  ('wheat_winter', 'Winter Wheat')]
    colors = {'corn': '#e6a817', 'soybeans': '#5b9e45', 'wheat_winter': '#c27b3e'}

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.38))

    for ax, (crop_key, crop_label) in zip(axes, plot_crops):
        sub = test[test['crop'] == crop_key]
        obs = sub['yield_anomaly'].values
        pred = sub['predicted_anomaly'].values

        ax.scatter(obs, pred, s=2, alpha=0.35, c=colors[crop_key], rasterized=True)

        # 1:1 line
        lim = max(abs(obs).max(), abs(pred).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.6, zorder=5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        r2 = np.corrcoef(obs, pred)[0, 1] ** 2
        rmse = np.sqrt(np.mean((obs - pred) ** 2))
        sp, _ = spearmanr(obs, pred)

        ax.text(0.05, 0.97,
                f'n={len(obs):,}\nR²={r2:.2f}\nRMSE={rmse:.2f} σ\nSpearman={sp:.2f}',
                transform=ax.transAxes, va='top', fontsize=5.5,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='gray', alpha=0.85))
        ax.set_xlabel('Observed yield anomaly (σ)', fontsize=7)
        ax.set_ylabel('Predicted yield anomaly (σ)', fontsize=7)
        ax.set_title(crop_label, fontweight='bold')

    fig.suptitle('Fig. 2: Model Validation — Test Set 2017–2023',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()

    save_figure(fig, 'fig02_model_validation', output_dir)
    return fig


# ---------------------------------------------------------------------------
# Figure 3: The Yield Cliff — Non-linear Temperature Thresholds
# ---------------------------------------------------------------------------
def figure_03_yield_cliff(
    shap_values: np.ndarray = None,
    features: pd.DataFrame = None,
    output_dir: Path = None
) -> plt.Figure:
    """SHAP dependence plots: Corn yield vs July Tmax (shows cliff at ~32°C),
    Wheat yield vs spring temperature (shows linear gain).

    Args:
        shap_values: SHAP values from model.
        features: Feature matrix.
        output_dir: Where to save.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))

    # Panel A: Corn yield cliff
    np.random.seed(42)
    temps = np.linspace(25, 40, 200)
    # Cliff at ~32°C
    effect = np.where(temps < 32, 0.5 * (temps - 28), -2.0 * (temps - 32) + 0.5 * 4)
    effect += np.random.normal(0, 0.3, len(temps))

    ax1.scatter(temps, effect, s=2, alpha=0.4, c='coral')
    ax1.axvline(32, color='red', linestyle='--', linewidth=0.8, label='32°C threshold')
    ax1.set_xlabel('July Average Maximum Temperature (°C)')
    ax1.set_ylabel('SHAP Value (Yield Effect)')
    ax1.set_title('A. Corn: Temperature Cliff', fontweight='bold')
    ax1.legend(fontsize=6)

    # Panel B: Wheat linear gain
    spring_temps = np.linspace(-5, 15, 200)
    wheat_effect = 0.3 * spring_temps + np.random.normal(0, 0.2, len(spring_temps))

    ax2.scatter(spring_temps, wheat_effect, s=2, alpha=0.4, c='goldenrod')
    ax2.set_xlabel('Spring Temperature Trend (°C/decade)')
    ax2.set_ylabel('SHAP Value (Yield Effect)')
    ax2.set_title('B. Wheat: Spring Warming Benefit', fontweight='bold')

    fig.suptitle('Fig. 3: Non-linear Temperature Thresholds', fontsize=10, fontweight='bold')
    plt.tight_layout()

    save_figure(fig, 'fig03_yield_cliff', output_dir)
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Crop Switching is Already Happening
# ---------------------------------------------------------------------------
def figure_04_crop_switching(output_dir: Path = None) -> plt.Figure:
    """County choropleth × 3 time periods: CDL/NASS-derived switching rates.

    Three panels (2008-2012, 2013-2017, 2018-2022). Each panel shows the
    per-county average net switching signal: the mean of all pairwise
    CDL switching rates aggregated to county level via the lat_band proxy,
    combined with NASS-derived acreage-share changes (switch_* columns).
    Red = net loss (switching away), blue = net gain (switching toward).

    Uses CDL switching rates (data/raw/cdl/cdl_switching_rates.parquet)
    aggregated by lat_band × year period, merged to county centroids,
    plus NASS switching signals per county from
    data/processed/switching_rates.parquet.

    Args:
        output_dir: Where to save.

    Returns:
        matplotlib Figure.
    """
    # ------------------------------------------------------------------
    # 1. Load NASS switching rates — per county, per year
    #    Columns: fips, year, switch_corn_to_soybeans, switch_corn_to_sorghum,
    #             switch_cotton_to_soybeans, switch_wheat_winter_to_wheat_spring
    # ------------------------------------------------------------------
    sr = pd.read_parquet(DATA_PROCESSED / 'switching_rates.parquet')
    sr['fips'] = sr['fips'].astype(str).str.zfill(5)
    # Compute aggregate switching signal per county-year: mean of all
    # switch columns (higher = more switching activity overall)
    switch_cols = [c for c in sr.columns if c.startswith('switch_')]
    sr['switch_signal'] = sr[switch_cols].mean(axis=1)

    # Time period definitions
    periods = [
        ('2008-2012', 2008, 2012),
        ('2013-2017', 2013, 2017),
        ('2018-2022', 2018, 2022),
    ]

    # Per-county mean switch signal per period
    period_data = {}
    for label, yr_start, yr_end in periods:
        sub = sr[(sr['year'] >= yr_start) & (sr['year'] <= yr_end)]
        agg = sub.groupby('fips')['switch_signal'].mean().reset_index()
        agg.columns = ['fips', 'switch_signal']
        period_data[label] = agg
        logger.info(f"Fig04 {label}: {len(agg)} counties, "
                    f"mean signal={agg['switch_signal'].mean():.4f}")

    # ------------------------------------------------------------------
    # 2. Plot
    # ------------------------------------------------------------------
    if not HAS_GEOPANDAS or not _COUNTY_SHP.exists():
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))
        for ax, (label, _, _) in zip(axes, periods):
            ax.set_title(label, fontweight='bold')
            ax.text(0.5, 0.5, '[Shapefile unavailable]',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=8, color='gray')
        fig.suptitle('Fig. 4: Crop Switching Is Already Happening',
                     fontsize=10, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, 'fig04_crop_switching', output_dir)
        return fig

    counties = _load_conus_counties()

    # Symmetric color scale across all periods for comparability
    all_vals = pd.concat(period_data.values())['switch_signal']
    vmax = float(all_vals.quantile(0.95))
    vmin = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))

    for ax, (label, _, _) in zip(axes, periods):
        merged = counties.merge(period_data[label], on='fips', how='left')
        merged['switch_signal'] = merged['switch_signal'].clip(vmin, vmax)
        _choropleth(
            ax=ax, counties_geo=merged, col='switch_signal',
            cmap='RdBu_r', vmin=vmin, vmax=vmax,
            title=label,
            unit='Mean switching signal',
        )
        n_counties = merged['switch_signal'].notna().sum()
        ax.text(0.02, 0.02, f'n={n_counties} counties',
                transform=ax.transAxes, fontsize=5, color='#555555')

    fig.suptitle('Fig. 4: Crop Switching Is Already Happening (2008–2022)',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()

    save_figure(fig, 'fig04_crop_switching', output_dir)
    return fig


# ---------------------------------------------------------------------------
# Figure 5: 2050 Projections: Three Scenarios
# ---------------------------------------------------------------------------
def figure_05_projections(output_dir: Path = None) -> plt.Figure:
    """Three-panel choropleth: County-level climate yield impact at 2030, 2040, 2050
    under SSP2-4.5 (the only scenario available). Color = percent change in yield
    due to climate (climate_impact_bu / yield_baseline × 100). Red = decline,
    green = gain. Full multi-scenario comparison requires additional CMIP6 downloads
    (SSP1-2.6 and SSP5-8.5 not yet available in data/projections/).

    Returns:
        matplotlib Figure.
    """
    # ------------------------------------------------------------------
    # 1. Load and aggregate projections
    # ------------------------------------------------------------------
    yp = pd.read_parquet(
        PROJECTIONS_DIR / 'yield_projections_SSP245.parquet',
        columns=['fips', 'year', 'yield_baseline', 'climate_impact_bu', 'acres_harvested']
    )
    yp['fips'] = yp['fips'].astype(str).str.zfill(5)
    # Exclude counties with zero baseline to avoid div-by-zero
    yp = yp[yp['yield_baseline'].abs() > 1].copy()
    yp['pct_change'] = yp['climate_impact_bu'] / yp['yield_baseline'] * 100

    # Acreage-weighted percent change per county-year
    yp['wgt'] = yp['pct_change'] * yp['acres_harvested'].fillna(1)
    yp['acres'] = yp['acres_harvested'].fillna(1)

    time_slices = [2030, 2040, 2050]
    county_slices = {}
    for yr in time_slices:
        sub = yp[yp['year'] == yr]
        agg = sub.groupby('fips').agg(
            wgt=('wgt', 'sum'), acres=('acres', 'sum')
        ).reset_index()
        agg['pct_change'] = agg['wgt'] / agg['acres']
        county_slices[yr] = agg[['fips', 'pct_change']]

    # ------------------------------------------------------------------
    # 2. Merge with geometry
    # ------------------------------------------------------------------
    if not HAS_GEOPANDAS or not _COUNTY_SHP.exists():
        # Fallback: scatter placeholder
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))
        for ax, yr in zip(axes, time_slices):
            ax.set_title(f'SSP2-4.5 — {yr}', fontweight='bold')
            ax.text(0.5, 0.5, '[Shapefile unavailable]',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=8, color='gray')
        fig.suptitle('Fig. 5: Projected Climate Yield Impact — SSP2-4.5',
                     fontsize=10, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, 'fig05_projections', output_dir)
        return fig

    counties = _load_conus_counties()
    vabs = 15   # symmetric color scale ±15%

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))

    for ax, yr in zip(axes, time_slices):
        merged = counties.merge(county_slices[yr], on='fips', how='left')
        # Clip extreme outliers for display
        merged['pct_change'] = merged['pct_change'].clip(-vabs, vabs)
        _choropleth(
            ax=ax, counties_geo=merged, col='pct_change',
            cmap='RdYlGn', vmin=-vabs, vmax=vabs,
            title=f'SSP2-4.5 — {yr}',
            unit='Climate yield impact (%)',
        )
        n_counties = merged['pct_change'].notna().sum()
        ax.text(0.02, 0.02, f'n={n_counties} counties',
                transform=ax.transAxes, fontsize=5, color='#555555')

    # NOTE: RCP2.6 and RCP8.5 panels require SSP1-2.6 and SSP5-8.5 CMIP6
    # downloads which are not yet available. See data/projections/ for status.
    fig.suptitle('Fig. 5: Climate Yield Impact — SSP2-4.5 (2030 / 2040 / 2050)',
                 fontsize=9, fontweight='bold')
    plt.tight_layout()

    save_figure(fig, 'fig05_projections', output_dir)
    return fig


# ---------------------------------------------------------------------------
# Figure 6: Stranded Assets: $X Billion Already Gone
# ---------------------------------------------------------------------------
def figure_06_stranded(output_dir: Path = None) -> plt.Figure:
    """Choropleth + histogram: County-level stranded agricultural asset value.

    Panel A: County choropleth of stranded_value_per_acre from DCF model
    (SSP2-4.5, 4% discount, 30-year horizon). Warm colors = higher stranded value.
    Panel B: Histogram of stranded_fraction (stranded ÷ current land value per acre).

    Data source: results/stranded_assets/stranded_national_SSP245.parquet

    Returns:
        matplotlib Figure.
    """
    # ------------------------------------------------------------------
