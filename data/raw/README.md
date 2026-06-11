# Raw data setup

Raw inputs are not tracked in git (~12 GB). Download once before running pipelines.

## Required files

| Path | Source |
|------|--------|
| `nass/nass_county_yields.parquet` | [USDA NASS QuickStats](https://quickstats.nass.usda.gov/) |
| `nass/nass_land_values.parquet` | NASS QuickStats |
| `prism/county_climate_*.parquet` | [PRISM / nClimDiv](https://www.ncei.noaa.gov/) |
| `rma/rma_sob_all_years.parquet` | [USDA RMA SOB](https://www.rma.usda.gov/SummaryOfBusiness) |
| `census/acs_*.parquet` | [Census ACS API](https://api.census.gov/) |
| `census/cc-est*.csv` | Census PEP (prime-age population) |
| `other/ers_atlas/` | [USDA ERS Atlas](https://www.ers.usda.gov/data-products/county-typology-codes/) |
| `other/cpi_annual.csv` | BLS CPI-U (2023 base = 304.7) |
| `../projections/*.parquet` | CMIP6 SSP2-4.5 / SSP3-7.0 (Pangeo) |

County yields: 1950–2023. Climate: monthly Tmax and precip.

## Processed outputs

`make ingest` and `make features` write:

- `data/processed/feature_matrix.parquet`
- `data/processed/county_panel.parquet`

These are gitignored and rebuilt locally.

## Published dataset

County-level CSVs and full tarball: see `data/published_dataset/README.md`.
