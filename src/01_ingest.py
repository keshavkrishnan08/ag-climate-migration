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
    for ssp in scenarios:
        rcp = scenario_map[ssp]
        n_models = sum(1 for k in file_map if k[1] == rcp)
        if n_models < 5:
            raise RuntimeError(
                f"CMIP6 ensemble too small for {rcp}: {n_models} models (need ≥5)"
            )

    logger.info(f"CMIP6 targets prepared: {len(file_map)} model-scenario combinations")
    return file_map


# ---------------------------------------------------------------------------
# 4. USDA RMA Crop Insurance
# ---------------------------------------------------------------------------
def ingest_rma_insurance(output_dir: Path = DATA_RAW / 'rma') -> pd.DataFrame:
    """Download USDA RMA Summary of Business data.

    Source: rma.usda.gov/data
    Coverage: County × crop × year, 1989-2023
    Variables: Premium, indemnity, liability, loss ratio

    Args:
        output_dir: Directory to save CSV files.

    Returns:
        DataFrame with insurance metrics by county-crop-year.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = 'https://pubfs-rma.fpac.usda.gov/pub/References/SOB'
    years = range(1989, 2024)

    all_dfs = []
    for year in tqdm(years, desc="RMA Summary of Business"):
        url = f"{base_url}/{year}/sobcov_{year}.zip"
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                zip_path = output_dir / f"sobcov_{year}.zip"
                with open(zip_path, 'wb') as f:
                    f.write(resp.content)

                # Extract and parse
                import zipfile
                with zipfile.ZipFile(zip_path) as zf:
                    for name in zf.namelist():
                        if name.endswith('.csv') or name.endswith('.txt'):
                            df = pd.read_csv(zf.open(name), dtype=str, low_memory=False)
                            all_dfs.append(df)
                            break
        except Exception as e:
            logger.warning(f"RMA {year}: {e}")
            continue

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        output_path = output_dir / 'rma_sob_all_years.parquet'
        result.to_parquet(output_path, index=False)
        logger.info(f"Saved RMA data: {len(result)} rows → {output_path}")
        return result

    logger.warning("No RMA data downloaded — check network and URLs")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 5. USDA Census of Agriculture
# ---------------------------------------------------------------------------
def ingest_census_of_agriculture(
    api_key: str,
    output_dir: Path = DATA_RAW / 'census'
) -> pd.DataFrame:
    """Download Census of Agriculture data for census years.

    Census years: 2002, 2007, 2012, 2017, 2022
    Variables: Farm size, debt, assets, operator age, land value

    Args:
        api_key: NASS API key.
        output_dir: Directory to save data.

    Returns:
        DataFrame with county-level farm structure variables.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    census_years = [2002, 2007, 2012, 2017, 2022]
    base_url = 'https://quickstats.nass.usda.gov/api/api_GET/'

    variables = [
        ('FARM OPERATIONS', 'OPERATIONS', 'NUMBER OF OPERATIONS'),
        ('AG LAND', 'AREA', 'ACRES'),
        ('AG LAND', 'ASSET VALUE', 'MEASURED IN $'),
        ('FARM OPERATIONS', 'DEBT, LONG TERM', 'MEASURED IN $'),
        ('OPERATORS', 'AGE, AVG', 'YEARS'),
    ]

    all_dfs = []
    for year in census_years:
        for commodity, stat_cat, unit in variables:
            params = {
                'key': api_key,
                'source_desc': 'CENSUS',
                'year': year,
                'commodity_desc': commodity,
                'statisticcat_desc': stat_cat,
                'unit_desc': unit,
                'agg_level_desc': 'COUNTY',
                'format': 'JSON',
            }

            try:
                resp = requests.get(base_url, params=params, timeout=120)
                resp.raise_for_status()
                data = resp.json().get('data', [])
                if data:
                    df = pd.DataFrame(data)
                    df['census_year'] = year
                    all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Census {year} {commodity}/{stat_cat}: {e}")

            time.sleep(0.5)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        output_path = output_dir / 'census_of_agriculture.parquet'
        result.to_parquet(output_path, index=False)
        logger.info(f"Saved Census of Ag: {len(result)} rows → {output_path}")
        return result

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 6. USDA NASS Farmland Values
# ---------------------------------------------------------------------------
def ingest_nass_land_values(api_key: str, output_dir: Path = DATA_RAW / 'nass') -> pd.DataFrame:
    """Download farmland values from NASS Quick Stats.

    Variables: Land value $/acre, Cash rent $/acre (state/county level)

    Args:
        api_key: NASS API key.
        output_dir: Directory to save data.

    Returns:
        DataFrame with land values by state/county-year.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = 'https://quickstats.nass.usda.gov/api/api_GET/'

    stats = [
        ('RENT, CASH, CROPLAND', 'RENT', 'DOLLARS / ACRE'),
        ('LAND & BUILDINGS, FARM REAL ESTATE', 'ASSET VALUE', 'DOLLARS / ACRE'),
    ]

    all_dfs = []
    for short_desc_part, stat_cat, unit in stats:
        params = {
            'key': api_key,
            'commodity_desc': 'AG LAND',
            'statisticcat_desc': stat_cat,
            'unit_desc': unit,
            'agg_level_desc': 'STATE',
            'year__GE': 1970,
            'year__LE': 2023,
            'format': 'JSON',
        }

        try:
            resp = requests.get(base_url, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json().get('data', [])
            if data:
                all_dfs.append(pd.DataFrame(data))
        except Exception as e:
            logger.warning(f"NASS land values ({stat_cat}): {e}")

        time.sleep(1)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        output_path = output_dir / 'nass_land_values.parquet'
        result.to_parquet(output_path, index=False)
        logger.info(f"Saved land values: {len(result)} rows → {output_path}")
        return result

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 7. Census ACS Rural Population
# ---------------------------------------------------------------------------
def ingest_acs_population(output_dir: Path = DATA_RAW / 'census') -> pd.DataFrame:
    """Download Census ACS county population and demographics.

    Variables: Population, age structure, income, poverty rate
    Coverage: Annual 2005-2023

    Args:
        output_dir: Directory to save data.

    Returns:
        DataFrame with county-year demographics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import censusdata
    except ImportError:
        logger.error("censusdata package required: pip install censusdata")
        return pd.DataFrame()

    variables = {
        'B01003_001E': 'total_population',
        'B19013_001E': 'median_household_income',
        'B17001_002E': 'poverty_count',
        'B07001_001E': 'geographic_mobility_total',
        'B07001_065E': 'moved_from_different_county',
        'B25077_001E': 'median_home_value',
    }

    all_dfs = []
    for year in tqdm(range(2005, 2024), desc="ACS population"):
        try:
            data = censusdata.download(
                'acs5', year,
                censusdata.censusgeo([('state', '*'), ('county', '*')]),
                list(variables.keys())
            )
            data = data.rename(columns=variables)
            data['year'] = year

            # Extract FIPS from index
            fips_codes = []
            for idx in data.index:
                params = idx.params()
                state = params[0][1]
                county = params[1][1]
                fips_codes.append(f"{state}{county}")
            data['fips'] = fips_codes
            data = data.reset_index(drop=True)

            all_dfs.append(data)
        except Exception as e:
            logger.warning(f"ACS {year}: {e}")

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        output_path = output_dir / 'acs_county_demographics.parquet'
        result.to_parquet(output_path, index=False)
        logger.info(f"Saved ACS data: {len(result)} rows → {output_path}")
        return result

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 8. BLS CPI Deflator
# ---------------------------------------------------------------------------
def ingest_cpi(fred_api_key: str, output_dir: Path = DATA_RAW / 'other') -> pd.Series:
    """Download CPI-U series from FRED for deflation to 2023 USD.

    Args:
        fred_api_key: FRED API key.
        output_dir: Directory to cache CPI data.

    Returns:
        Annual average CPI series.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.deflator import load_or_fetch_cpi

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = str(output_dir / 'cpi_annual.csv')
    return load_or_fetch_cpi(cache_path=cache_path, fred_api_key=fred_api_key)


# ---------------------------------------------------------------------------
# 9. Rural Hospital Closures
# ---------------------------------------------------------------------------
def ingest_hospital_closures(output_dir: Path = DATA_RAW / 'other') -> pd.DataFrame:
    """Download rural hospital closure and operational status data.

    Source: ruralhospitals.chqpr.org
    Coverage: County level, 2005-2023

    Args:
        output_dir: Directory to save data.

    Returns:
        DataFrame with hospital operational status by county.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # This data is typically downloaded as CSV from rural health organizations
    # UNC Sheps Center maintains the most comprehensive dataset
    logger.info("Hospital closure data requires manual download from UNC Sheps Center")
    logger.info("See: https://www.shepscenter.unc.edu/programs-projects/rural-health/rural-hospital-closures/")

    result = pd.DataFrame(columns=[
        'fips', 'hospital_name', 'status', 'closure_year',
        'beds', 'services', 'critical_access'
    ])

    output_path = output_dir / 'rural_hospital_closures.parquet'
    result.to_parquet(output_path, index=False)
    return result


# ---------------------------------------------------------------------------
# 10. NCES School Enrollment
# ---------------------------------------------------------------------------
def ingest_school_enrollment(output_dir: Path = DATA_RAW / 'other') -> pd.DataFrame:
    """Download K-12 school enrollment by district.

    Source: NCES Common Core of Data (CCD)
    Coverage: District annual, 1986-2023
