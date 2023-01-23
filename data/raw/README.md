# Raw Data Setup

Raw inputs are **not** tracked in git (see `.gitignore`). Download once (~12 GB; ~3 hours) before running the pipeline.

## Required files

| File / directory | Source | Notes |
|------------------|--------|-------|
| `nass/nass_county_yields.parquet` | [USDA NASS QuickStats](https://quickstats.nass.usda.gov/) | County yield + acreage, 1950–2023 |
| `nass/nass_land_values.parquet` | NASS QuickStats | Farmland value per acre |
