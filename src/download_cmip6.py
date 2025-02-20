"""Stream-download CMIP6 data from NASA NEX-GDDP-CMIP6 (public S3).

Downloads one file at a time, extracts CONUS, aggregates to county
monthly means, deletes raw file. Peak disk: ~250 MB. Final output: ~350 MB.

Usage:
    python src/download_cmip6.py                    # ssp245 only (primary)
    python src/download_cmip6.py --all-scenarios    # ssp126 + ssp245 + ssp585
"""

