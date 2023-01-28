"""County FIPS utilities for CONUS agricultural analysis."""

import pandas as pd
import numpy as np
from loguru import logger

# State FIPS codes to exclude (Alaska, Hawaii, Puerto Rico)
EXCLUDED_STATE_FIPS = {'02', '15', '72'}

# All 50 states + DC FIPS codes (CONUS = exclude AK, HI)
