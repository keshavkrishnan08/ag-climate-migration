"""CPI deflation to 2023 USD using BLS CPI-U series."""

import pandas as pd
import numpy as np
from loguru import logger

try:
    from fredapi import Fred
except ImportError:
    Fred = None
