"""Phase 1: Data acquisition — all 12 datasets.

Downloads, validates, and caches all raw data needed for the pipeline.
Data sources per PRD Section 3.1:
    1. USDA NASS County Yields (API)
    2. USDA NASS Cropland Data Layer (GDAL raster)
    3. PRISM Climate Data (API)
    4. CMIP6 Climate Projections (ESGF Python API)
    5. USDA RMA Summary of Business (CSV)
    6. USDA Census of Agriculture (API + CSV)
