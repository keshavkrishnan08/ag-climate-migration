"""
Build county-level climate projections from CMIP6 SSP3-7.0 gridded data.

Uses the same delta method as build_county_climate_projections.py but reads
from data/raw/cmip6_ssp370/ and writes to
data/projections/county_climate_projections_ssp370.parquet.

Units, delta method, and interpolation logic are identical to the SSP2-4.5 version.
GCM substitutions vs. SSP2-4.5:
  - MPI-ESM1-2-HR -> MPI-ESM1-2-LR (HR not in Pangeo ssp370)
