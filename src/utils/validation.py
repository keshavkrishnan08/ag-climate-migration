"""Cross-validation helpers with strict temporal ordering."""

import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Tuple, Generator


def temporal_rolling_cv(
    years: np.ndarray,
    n_folds: int = 5,
