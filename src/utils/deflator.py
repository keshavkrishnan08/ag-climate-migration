"""CPI deflation to 2023 USD using BLS CPI-U series."""

import pandas as pd
import numpy as np
from loguru import logger

try:
    from fredapi import Fred
except ImportError:
    Fred = None


def fetch_cpi_series(fred_api_key: str, series_id: str = 'CPIAUCSL') -> pd.Series:
    """Fetch CPI-U monthly series from FRED.

    Args:
        fred_api_key: FRED API key string.
        series_id: FRED series identifier for CPI.

    Returns:
        Monthly CPI-U index as pandas Series with datetime index.
    """
    if Fred is None:
        raise ImportError("fredapi package required. Install via: pip install fredapi")
    fred = Fred(api_key=fred_api_key)
    cpi = fred.get_series(series_id)
    logger.info(f"Fetched {len(cpi)} CPI observations from FRED ({series_id})")
    return cpi


def build_annual_cpi(cpi_monthly: pd.Series) -> pd.Series:
    """Convert monthly CPI to annual averages.

    Args:
        cpi_monthly: Monthly CPI-U series with datetime index.

    Returns:
        Annual average CPI indexed by year (int).
    """
    annual = cpi_monthly.groupby(cpi_monthly.index.year).mean()
    annual.index.name = 'year'
    return annual


def deflate_to_2023(
    values: np.ndarray,
    years: np.ndarray,
    cpi_annual: pd.Series,
    base_year: int = 2023
) -> np.ndarray:
    """Deflate nominal dollar values to base-year (2023) USD.

    Args:
        values: Array of nominal dollar amounts.
        years: Array of corresponding years (same length as values).
        cpi_annual: Annual average CPI series indexed by year.
        base_year: Target year for constant dollars.

    Returns:
        Array of deflated values in base_year USD.
