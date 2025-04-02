"""
Download CMIP6 SSP3-7.0 data from Pangeo Cloud for 9 GCMs.

SSP3-7.0 eliminates the single-scenario limitation in the AgMigration pipeline.
Uses anonymous GCS access to the Pangeo CMIP6 zarr archive — no ESGF account needed.

Availability notes vs. SSP2-4.5:
  - MPI-ESM1-2-HR: not in Pangeo ssp370; substituted with MPI-ESM1-2-LR (same institution)
  - NorESM2-MM: ssp370 data exists in Pangeo but only has tas/pr, no tasmax/tasmin
  - HadGEM3-GC31-LL: not in Pangeo ssp370 at all
