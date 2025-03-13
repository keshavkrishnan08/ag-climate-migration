"""
Download 5 missing CMIP6 GCMs from Pangeo Cloud (anonymous GCS access).

Models: CESM2, CNRM-CM6-1, HadGEM3-GC31-LL, IPSL-CM6A-LR, MRI-ESM2-0
Scenario: SSP2-4.5
Variables: tasmax, tasmin, pr
Years: 2025-2050 (extracted from 2015-2100 zarr stores)

Outputs:
    data/raw/cmip6/{MODEL}_ssp245_{VAR}_{YEAR}_conus_monthly.parquet
