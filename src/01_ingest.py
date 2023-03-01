"""Phase 1: Data acquisition — all 12 datasets.

Downloads, validates, and caches all raw data needed for the pipeline.
Data sources per PRD Section 3.1:
    1. USDA NASS County Yields (API)
    2. USDA NASS Cropland Data Layer (GDAL raster)
    3. PRISM Climate Data (API)
    4. CMIP6 Climate Projections (ESGF Python API)
    5. USDA RMA Summary of Business (CSV)
    6. USDA Census of Agriculture (API + CSV)
    7. USDA NASS Farmland Values (API)
    8. Census ACS Rural Population (API)
    9. BLS CPI Deflator (FRED API)
    10. Rural Hospital Closures (CSV)
    11. NCES School Enrollment (CSV)
    12. USDA GIPSA Grain Elevators (CSV)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from loguru import logger
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

with open(PROJECT_ROOT / 'config.yaml') as f:
    CONFIG = yaml.safe_load(f)

CROPS = CONFIG['crops']['primary']
FIPS_EXCLUDE = CONFIG['geography']['fips_exclude']


# ---------------------------------------------------------------------------
# 1. USDA NASS County Yields
# ---------------------------------------------------------------------------
def ingest_nass_yields(api_key: str, output_dir: Path = DATA_RAW / 'nass') -> pd.DataFrame:
    """Download county-level crop yields from USDA NASS Quick Stats API.

    Args:
        api_key: NASS API key (request at https://quickstats.nass.usda.gov/api).
        output_dir: Directory to save raw CSV files.

    Returns:
        DataFrame with columns: fips, year, crop, yield_bu_acre, acres_harvested, production.

    Raises:
        requests.HTTPError: If API request fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = 'https://quickstats.nass.usda.gov/api/api_GET/'

    nass_crop_names = {
        'corn': 'CORN',
        'soybeans': 'SOYBEANS',
        'wheat_winter': 'WHEAT, WINTER',
        'wheat_spring': 'WHEAT, SPRING, (EXCL DURUM)',
        'cotton': 'COTTON, UPLAND',
        'sorghum': 'SORGHUM, GRAIN',
        'barley': 'BARLEY',
        'oats': 'OATS',
    }

    all_dfs = []
    for crop_key, nass_name in nass_crop_names.items():
        logger.info(f"Fetching NASS yields for {crop_key} ({nass_name})")

        params = {
            'key': api_key,
            'commodity_desc': nass_name.split(',')[0].strip(),
            'statisticcat_desc': 'YIELD',
            'unit_desc': 'BU / ACRE',
            'agg_level_desc': 'COUNTY',
            'year__GE': 1950,
            'year__LE': 2023,
            'format': 'JSON',
        }
        if ',' in nass_name:
            params['short_desc'] = f"{nass_name} - YIELD, MEASURED IN BU / ACRE"

        resp = requests.get(base_url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json().get('data', [])

        if not data:
            logger.warning(f"No data returned for {crop_key}")
            continue

        df = pd.DataFrame(data)
        df = df[['state_fips_code', 'county_code', 'year', 'Value']].copy()
        df['fips'] = df['state_fips_code'].astype(str).str.zfill(2) + df['county_code'].astype(str).str.zfill(3)
        df['year'] = df['year'].astype(int)
        df['yield_bu_acre'] = pd.to_numeric(df['Value'].str.replace(',', ''), errors='coerce')
        df['crop'] = crop_key
        df = df[['fips', 'year', 'crop', 'yield_bu_acre']].dropna(subset=['yield_bu_acre'])

        all_dfs.append(df)
        logger.info(f"  {crop_key}: {len(df)} county-year observations")
        time.sleep(1)  # Rate limit
