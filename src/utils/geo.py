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
    df[fips_col] = df[fips_col].astype(str).str.zfill(5)
    state_fips = df[fips_col].str[:2]
    mask = ~state_fips.isin(EXCLUDED_STATE_FIPS)
    n_excluded = (~mask).sum()
    if n_excluded > 0:
        logger.info(f"Excluded {n_excluded} non-CONUS records (AK/HI/PR)")
    return df[mask].reset_index(drop=True)


def validate_fips_coverage(
    df: pd.DataFrame,
    fips_col: str = 'fips',
    expected_n: int = 3108
) -> dict:
    """Check that we have the expected number of CONUS counties.

    Args:
        df: DataFrame to validate.
        fips_col: Column containing FIPS codes.
        expected_n: Expected number of unique counties.

    Returns:
        Dict with 'n_counties', 'missing_fraction', 'passed' keys.
    """
    unique_fips = df[fips_col].astype(str).str.zfill(5).nunique()
    missing_frac = 1.0 - (unique_fips / expected_n)

    result = {
        'n_counties': unique_fips,
        'expected': expected_n,
        'missing_fraction': missing_frac,
        'passed': unique_fips >= expected_n * 0.95  # allow 5% missing
    }

    if result['passed']:
        logger.info(f"FIPS coverage check PASSED: {unique_fips}/{expected_n} counties")
    else:
        logger.warning(f"FIPS coverage check FAILED: {unique_fips}/{expected_n} counties ({missing_frac:.1%} missing)")

    return result
