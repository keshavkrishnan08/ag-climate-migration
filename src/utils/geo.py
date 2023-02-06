"""County FIPS utilities for CONUS agricultural analysis."""

import pandas as pd
import numpy as np
from loguru import logger

# State FIPS codes to exclude (Alaska, Hawaii, Puerto Rico)
EXCLUDED_STATE_FIPS = {'02', '15', '72'}

# All 50 states + DC FIPS codes (CONUS = exclude AK, HI)
CONUS_STATE_FIPS = {
    f'{i:02d}' for i in range(1, 57)
    if f'{i:02d}' not in EXCLUDED_STATE_FIPS and i not in (3, 7, 14, 43, 52)
}


def load_county_fips(path: str = None) -> pd.DataFrame:
    """Load complete county FIPS table.

    Args:
