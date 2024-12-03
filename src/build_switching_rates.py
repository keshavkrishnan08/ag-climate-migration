"""Build county-level crop switching rates from NASS acreage data.

Computes year-over-year share changes as a proxy for crop switching.
For each pair (A→B): when A's share drops >5pp AND B's share rises,
the switching rate equals the increase in B's acreage share.

Output: data/processed/switching_rates.parquet
Columns:
    fips                          (str, 5-digit zero-padded)
    year                          (int)
    switch_corn_to_soybeans       (float, 0-1)
    switch_corn_to_sorghum        (float, 0-1)
    switch_cotton_to_soybeans     (float, 0-1)
    switch_wheat_winter_to_wheat_spring (float, 0-1)
"""

import sys
from pathlib import Path

import numpy as np
