# Raw Data Setup

Raw inputs are **not** tracked in git (see `.gitignore`). Download once (~12 GB; ~3 hours) before running the pipeline.

## Required files

| File / directory | Source | Notes |
|------------------|--------|-------|
| `nass/nass_county_yields.parquet` | [USDA NASS QuickStats](https://quickstats.nass.usda.gov/) | County yield + acreage, 1950–2023 |
| `nass/nass_land_values.parquet` | NASS QuickStats | Farmland value per acre |
| `prism/county_climate_*.parquet` | [PRISM / nClimDiv](https://www.ncei.noaa.gov/) | Monthly Tmax, precip |
| `rma/rma_sob_all_years.parquet` | [USDA RMA SOB](https://www.rma.usda.gov/SummaryOfBusiness) | Crop insurance premiums |
| `census/acs_*.parquet` | [Census ACS API](https://api.census.gov/) | Demographics, migration |
| `census/cc-est*.csv` | Census PEP | Prime-age population |
| `other/ers_atlas/` | [USDA ERS Atlas](https://www.ers.usda.gov/data-products/county-typology-codes/) | Farming-dependent counties |
| `other/cpi_annual.csv` | BLS CPI-U | Deflate to 2023 USD (304.7) |
| `../projections/*.parquet` | CMIP6 via Pangeo | SSP2-4.5 / SSP3-7.0 downscaled |

## Processed intermediates

Running `make ingest features` writes:

- `data/processed/feature_matrix.parquet`
- `data/processed/county_panel.parquet`

These are regenerated locally and also gitignored.

## Zenodo archive

At acceptance, a single tarball with all inputs and the published county CSVs will receive a DOI (see `data/published_dataset/README.md`).
