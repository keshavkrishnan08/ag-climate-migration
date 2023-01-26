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


