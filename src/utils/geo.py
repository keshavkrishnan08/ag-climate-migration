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
        path: Path to county FIPS CSV. If None, builds from Census data.

    Returns:
        DataFrame with columns: fips (str, 5-digit), state_fips (str, 2-digit),
        county_fips (str, 3-digit), state_name, county_name.
    """
    if path is not None:
        df = pd.read_csv(path, dtype=str)
        df['fips'] = df['fips'].str.zfill(5)
        logger.info(f"Loaded {len(df)} county FIPS codes from {path}")
