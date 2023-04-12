"""Phase 2: Feature engineering — yields + climate + switching.

Builds the complete feature matrix for all county-crop-year observations.
Works with actual downloaded data:
    - NASS yields: data/raw/nass/nass_county_yields.parquet
    - Climate: data/raw/prism/county_climate_annual.parquet (NOAA nClimDiv, °F)
    - ACS demographics: data/raw/census/acs_county_demographics.parquet
    - Farm operations: data/raw/nass/nass_farm_operations.parquet
    - ERS Atlas: data/raw/other/ers_atlas/*.parquet
"""
