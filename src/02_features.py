"""Phase 2: Feature engineering — yields + climate + switching.

Builds the complete feature matrix for all county-crop-year observations.
Works with actual downloaded data:
    - NASS yields: data/raw/nass/nass_county_yields.parquet
    - Climate: data/raw/prism/county_climate_annual.parquet (NOAA nClimDiv, °F)
    - ACS demographics: data/raw/census/acs_county_demographics.parquet
    - Farm operations: data/raw/nass/nass_farm_operations.parquet
    - ERS Atlas: data/raw/other/ers_atlas/*.parquet
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

F_TO_C = lambda f: (f - 32) * 5 / 9


# ---------------------------------------------------------------------------
# Core derived variables
# ---------------------------------------------------------------------------
def compute_gdd_from_monthly(
    tmax_f: float,
    tmin_f: float,
    base_c: float,
    upper_c: float,
    days_in_month: int = 30
) -> float:
    """Approximate monthly GDD from monthly avg tmax/tmin (given in °F).

    Args:
        tmax_f: Monthly average max temperature in °F.
        tmin_f: Monthly average min temperature in °F.
        base_c: Crop base temperature in °C.
        upper_c: Crop upper threshold in °C.
        days_in_month: Days in the month.

    Returns:
        Approximate GDD for that month.
    """
    if np.isnan(tmax_f) or np.isnan(tmin_f):
        return np.nan
    tmax_c = F_TO_C(tmax_f)
    tmin_c = F_TO_C(tmin_f)
    tavg = (tmax_c + tmin_c) / 2.0
    effective = min(tavg, upper_c)
    daily_gdd = max(0.0, effective - base_c)
    return daily_gdd * days_in_month


def compute_yield_anomaly(yields_series: pd.Series) -> pd.Series:
    """Remove technology trend from yield to isolate climate signal.

    Fits linear + quadratic trend. Returns z-score residuals.

    Args:
        yields_series: County-crop yields indexed by year.

    Returns:
        Detrended yield anomaly series (z-score).
    """
    s = yields_series.dropna()
    if len(s) < 10:
        return pd.Series(np.nan, index=yields_series.index)

    years = s.index.values.astype(float)
    values = s.values.astype(float)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        coeffs = np.polyfit(years, values, deg=2)
    trend = np.polyval(coeffs, years)
    residuals = values - trend

    std = residuals.std()
    anomaly = residuals / std if std > 0 else residuals
    return pd.Series(anomaly, index=s.index)


# ---------------------------------------------------------------------------
# Load real data files
# ---------------------------------------------------------------------------
def load_nass_yields() -> pd.DataFrame:
    """Load NASS county yields, filter to 1950+ with valid yields.

    Deduplicates to one record per county-crop-year (NASS bulk has
    multiple records per group for different practices/coverage types).

    Returns:
        DataFrame: fips, year, crop, yield_bu_acre, acres_harvested.
    """
    path = DATA_RAW / 'nass' / 'nass_county_yields.parquet'
    df = pd.read_parquet(path)
    df = df[df['year'] >= 1950].copy()
    df = df.dropna(subset=['yield_bu_acre'])
