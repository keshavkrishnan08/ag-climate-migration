"""Targeted CMIP6 download — only what the paper needs.

Downloads 5 diverse GCMs x ssp245 x 3 vars x milestone years.
Then linearly interpolates annual values for the projection pipeline.

Usage:
    python src/download_cmip6_targeted.py
"""

import sys
