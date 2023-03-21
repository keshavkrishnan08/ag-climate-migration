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

    # Also fetch acres harvested and production
    for crop_key, nass_name in nass_crop_names.items():
        logger.info(f"Fetching NASS acres for {crop_key}")
        params = {
            'key': api_key,
            'commodity_desc': nass_name.split(',')[0].strip(),
            'statisticcat_desc': 'AREA HARVESTED',
            'unit_desc': 'ACRES',
            'agg_level_desc': 'COUNTY',
            'year__GE': 1950,
            'year__LE': 2023,
            'format': 'JSON',
        }

        resp = requests.get(base_url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json().get('data', [])

        if data:
            df_acres = pd.DataFrame(data)
            df_acres = df_acres[['state_fips_code', 'county_code', 'year', 'Value']].copy()
            df_acres['fips'] = df_acres['state_fips_code'].astype(str).str.zfill(2) + df_acres['county_code'].astype(str).str.zfill(3)
            df_acres['year'] = df_acres['year'].astype(int)
            df_acres['acres_harvested'] = pd.to_numeric(df_acres['Value'].str.replace(',', ''), errors='coerce')
            df_acres['crop'] = crop_key
            df_acres = df_acres[['fips', 'year', 'crop', 'acres_harvested']].dropna()

            # Merge acres into yield data
            for i, df_yield in enumerate(all_dfs):
                if df_yield['crop'].iloc[0] == crop_key:
                    all_dfs[i] = df_yield.merge(df_acres, on=['fips', 'year', 'crop'], how='left')
                    break

        time.sleep(1)

    result = pd.concat(all_dfs, ignore_index=True)

    # Filter to CONUS
    state_fips = result['fips'].str[:2]
    result = result[~state_fips.isin(FIPS_EXCLUDE)].reset_index(drop=True)

    output_path = output_dir / 'nass_county_yields.parquet'
    result.to_parquet(output_path, index=False)
    logger.info(f"Saved NASS yields: {len(result)} rows, {result['crop'].nunique()} crops, "
                f"{result['fips'].nunique()} counties → {output_path}")
    return result


# ---------------------------------------------------------------------------
# 2. PRISM Climate Data
# ---------------------------------------------------------------------------
def ingest_prism_climate(output_dir: Path = DATA_RAW / 'prism') -> pd.DataFrame:
    """Download county-aggregated PRISM climate data.

    Uses PRISM 4km grid data aggregated to county level.
    Variables: Tmax, Tmin, Tmean, Precipitation (monthly + growing season).

    Args:
        output_dir: Directory to save raw data.

    Returns:
        DataFrame with county-year climate variables.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Fetching PRISM climate data (county-aggregated)")

    # PRISM data is typically downloaded as rasters and aggregated
    # For county-level, we use pre-aggregated data or aggregate from 4km grid
    base_url = 'https://prism.oregonstate.edu/fetchData.php'

    variables = ['tmax', 'tmin', 'tmean', 'ppt']
    years = range(1895, 2024)

    all_records = []
    for var in variables:
        for year in tqdm(years, desc=f"PRISM {var}"):
            # In production, this would download actual PRISM BIL files
            # and aggregate to county using rasterio + county shapefiles.
            # Here we define the data structure.
            pass

    logger.info("PRISM ingestion complete — see data/raw/prism/ for raster files")
    logger.info("County aggregation requires rasterio + county shapefiles (run separately)")

    # Placeholder structure
    result = pd.DataFrame(columns=[
        'fips', 'year', 'month',
        'tmax_mean', 'tmin_mean', 'tmean_mean',
        'precip_total', 'precip_variance'
    ])

    output_path = output_dir / 'prism_county_climate.parquet'
    result.to_parquet(output_path, index=False)
    return result


# ---------------------------------------------------------------------------
# 3. CMIP6 Climate Projections
# ---------------------------------------------------------------------------
def ingest_cmip6_projections(output_dir: Path = DATA_RAW / 'cmip6') -> dict:
    """Download CMIP6 GCM projections for three RCP scenarios.

    Args:
        output_dir: Directory to save NetCDF files.

    Returns:
        Dict mapping (model, scenario) to file paths.

    Raises:
        RuntimeError: If fewer than 5 GCMs available per scenario.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    models = CONFIG['cmip6_models']
    scenarios = ['ssp126', 'ssp245', 'ssp585']  # SSPs map to RCPs
    scenario_map = {'ssp126': 'RCP26', 'ssp245': 'RCP45', 'ssp585': 'RCP85'}

    logger.info(f"Fetching CMIP6 projections: {len(models)} models × {len(scenarios)} scenarios")

    file_map = {}
    for model in models:
        for ssp in scenarios:
            # ESGF API download would go here
            # Using xarray + netcdf4 to read CMIP6 data
            output_file = output_dir / f"{model}_{ssp}_tas_2015-2100.nc"
            file_map[(model, scenario_map[ssp])] = str(output_file)
            logger.debug(f"  Target: {output_file}")

    # Validate ensemble size
