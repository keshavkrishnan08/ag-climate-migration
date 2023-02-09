"""Cross-validation helpers with strict temporal ordering."""

import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Tuple, Generator


def temporal_rolling_cv(
    years: np.ndarray,
    n_folds: int = 5,
    train_start: int = 1950,
    val_window: int = 5
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate temporal rolling cross-validation splits.

    NEVER shuffles — strict temporal ordering always.

    Fold structure (from PRD Section 5.1):
        Fold 1: train 1950-1985, val 1986-1990
        Fold 2: train 1950-1990, val 1991-1995
        Fold 3: train 1950-1995, val 1996-2000
