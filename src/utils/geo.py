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
        return df

    try:
        import censusdata
        counties = censusdata.geographies(
            censusdata.censusgeo([('state', '*'), ('county', '*')]),
            'acs5', 2022
        )
        records = []
        for geo, name in counties.items():
            state_fips = geo.params()[0][1]
            county_fips = geo.params()[1][1]
            fips = state_fips + county_fips
            records.append({
                'fips': fips,
                'state_fips': state_fips,
                'county_fips': county_fips,
                'name': name
            })
        df = pd.DataFrame(records)
        logger.info(f"Built {len(df)} county FIPS codes from Census API")
        return df
    except Exception as e:
        logger.warning(f"Could not fetch from Census API: {e}")
        raise


def get_state_from_fips(fips: str) -> str:
    """Extract 2-digit state FIPS from 5-digit county FIPS.

    Args:
        fips: 5-digit county FIPS code as string.

    Returns:
        2-digit state FIPS code.
    """
    return str(fips).zfill(5)[:2]


def filter_conus_counties(df: pd.DataFrame, fips_col: str = 'fips') -> pd.DataFrame:
    """Filter DataFrame to only CONUS counties (exclude AK, HI, PR).

    Args:
        df: DataFrame containing county FIPS codes.
        fips_col: Name of the column containing 5-digit FIPS codes.

    Returns:
        Filtered DataFrame with only CONUS counties.
    """
    df = df.copy()
