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
        Fold 4: train 1950-2000, val 2001-2005
        Fold 5: train 1950-2005, val 2006-2010

    Args:
        years: Array of years for each observation.
        n_folds: Number of CV folds.
        train_start: First year of training data.
        val_window: Number of years in each validation window.

    Yields:
        Tuples of (train_indices, val_indices) for each fold.
    """
    years = np.asarray(years)
    val_starts = [1986, 1991, 1996, 2001, 2006]

    if n_folds != 5:
        max_val_year = years.max() - val_window
        val_starts = np.linspace(
            train_start + 30, max_val_year, n_folds
        ).astype(int).tolist()

    for fold_idx, val_start in enumerate(val_starts):
        val_end = val_start + val_window - 1
        train_mask = (years >= train_start) & (years < val_start)
        val_mask = (years >= val_start) & (years <= val_end)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            logger.warning(f"Fold {fold_idx + 1}: empty split (train={train_mask.sum()}, val={val_mask.sum()})")
            continue

        logger.info(
            f"Fold {fold_idx + 1}/{n_folds}: "
            f"train {train_start}-{val_start - 1} (n={len(train_idx)}), "
            f"val {val_start}-{val_end} (n={len(val_idx)})"
        )

        # Verify no future leakage
        assert years[train_idx].max() < years[val_idx].min(), \
            f"Temporal leakage detected in fold {fold_idx + 1}!"

        yield train_idx, val_idx


def check_no_future_leakage(
    train_years: np.ndarray,
    val_years: np.ndarray,
    test_years: np.ndarray = None
) -> bool:
    """Verify strict temporal ordering between train/val/test sets.

    Args:
        train_years: Years in training set.
        val_years: Years in validation set.
        test_years: Years in test set (optional).

    Returns:
        True if no leakage detected.

    Raises:
        AssertionError: If temporal ordering is violated.
    """
    train_max = np.max(train_years)
    val_min = np.min(val_years)
    val_max = np.max(val_years)

    assert train_max < val_min, \
        f"Train/val leakage: train max year ({train_max}) >= val min year ({val_min})"

    if test_years is not None:
        test_min = np.min(test_years)
        assert val_max < test_min, \
            f"Val/test leakage: val max year ({val_max}) >= test min year ({test_min})"

    logger.info("Temporal leakage check PASSED")
    return True


def compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    crop_name: str = ""
) -> dict:
    """Compute regression performance metrics.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        crop_name: Optional crop name for logging.

    Returns:
        Dict with rmse, mae, r2, spearman_rank keys.
    """
    from scipy import stats

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    spearman_corr, spearman_p = stats.spearmanr(y_true, y_pred)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'spearman_rank': spearman_corr,
        'spearman_p': spearman_p,
        'n_obs': len(y_true)
    }

    prefix = f"[{crop_name}] " if crop_name else ""
    logger.info(
        f"{prefix}RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}, "
        f"Spearman ρ={spearman_corr:.3f} (n={len(y_true)})"
    )

    return metrics
