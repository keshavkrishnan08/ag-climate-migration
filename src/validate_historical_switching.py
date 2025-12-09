"""Fix 5 — Validate crop switching model against 4 historical events.

Uses NASS county acreage data (1950+) and nClimDiv climate data to test
whether a simple climate-feature LightGBM model can reproduce observed
crop switching patterns. Four validation events from SWARM spec §A3:

1. Sorghum expansion in southern Plains (1950-1975) — POSITIVE
2. Cotton retreat from Missouri/Tennessee (1980-2010) — POSITIVE
3. Winter wheat boundary shift in Kansas (1990-2010) — POSITIVE
4. Soybean adoption in Corn Belt (1960-1980) — NEGATIVE test

All four must pass before the switching model can be trusted for projections.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NASS_PATH = PROJECT_ROOT / "data" / "raw" / "nass" / "nass_county_yields.parquet"
CLIMATE_PATH = PROJECT_ROOT / "data" / "raw" / "prism" / "county_climate_annual.parquet"
CLIMATE_MONTHLY_PATH = PROJECT_ROOT / "data" / "raw" / "prism" / "county_climate_monthly.parquet"
OUTPUT_PATH = PROJECT_ROOT / "state" / "validation" / "historical_switching.json"

RANDOM_SEED = 42

# Kansas county centroid latitudes (FIPS -> latitude in degrees N).
# Source: US Census Bureau TIGER/Gazetteer county centroids.
KANSAS_COUNTY_LAT = {
    "20001": 37.88, "20003": 37.77, "20005": 38.91, "20007": 37.25,
    "20009": 38.79, "20011": 38.48, "20013": 37.25, "20015": 37.80,
    "20017": 37.79, "20019": 37.27, "20021": 38.47, "20023": 37.56,
    "20025": 38.06, "20027": 37.56, "20029": 39.78, "20031": 37.24,
    "20033": 38.48, "20035": 37.25, "20037": 38.07, "20039": 39.05,
    "20041": 39.35, "20043": 38.68, "20045": 38.53, "20047": 38.68,
    "20049": 39.79, "20051": 39.35, "20053": 37.77, "20055": 38.26,
    "20057": 38.52, "20059": 37.24, "20061": 38.69, "20063": 38.07,
    "20065": 39.78, "20067": 38.07, "20069": 39.33, "20071": 37.56,
    "20073": 37.22, "20075": 37.86, "20077": 37.88, "20079": 37.80,
    "20081": 38.68, "20083": 37.56, "20085": 38.48, "20087": 38.92,
    "20089": 39.79, "20091": 38.49, "20093": 38.07, "20095": 38.26,
    "20097": 39.34, "20099": 38.50, "20101": 37.88, "20103": 37.58,
    "20105": 38.94, "20107": 38.06, "20109": 39.78, "20111": 39.06,
    "20113": 37.55, "20115": 37.86, "20117": 39.79, "20119": 37.26,
    "20121": 38.26, "20123": 39.05, "20125": 38.92, "20127": 38.92,
    "20129": 38.66, "20131": 37.26, "20133": 39.83, "20135": 38.92,
    "20137": 37.57, "20139": 39.05, "20141": 38.26, "20143": 39.04,
    "20145": 37.25, "20147": 38.06, "20149": 38.27, "20151": 37.24,
    "20153": 38.91, "20155": 39.35, "20157": 37.65, "20159": 37.26,
    "20161": 39.05, "20163": 39.34, "20165": 39.79, "20167": 38.86,
    "20169": 37.66, "20171": 39.06, "20173": 37.68, "20175": 37.25,
    "20177": 38.65, "20179": 37.55, "20181": 37.88, "20183": 39.04,
    "20185": 38.68, "20187": 37.80, "20189": 37.26, "20191": 37.56,
    "20193": 38.91, "20195": 37.80, "20197": 39.33, "20199": 38.06,
    "20201": 39.06, "20203": 39.79, "20205": 38.87, "20207": 38.68,
    "20209": 39.11,
}


# -----------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------

def load_nass_acreage(
    state_fips: list[str],
    crops: list[str] | None = None,
    year_min: int = 1950,
    year_max: int = 2025,
) -> pd.DataFrame:
    """Load deduplicated NASS county acreage data for target states/crops.

    Takes the MAX of acres_harvested per (fips, year, crop) to collapse
    the survey-item cross-product artifact in the raw NASS bulk download.

    Args:
        state_fips: 2-digit state FIPS strings to include.
        crops: Crop names to filter. None means all.
        year_min: Start year (inclusive).
        year_max: End year (inclusive).

    Returns:
        DataFrame with columns [fips, year, crop, acres_harvested],
        one row per county-year-crop.
    """
    df = pd.read_parquet(
        NASS_PATH,
        columns=["fips", "year", "crop", "acres_harvested"],
    )
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["fips"].str[:2]

    # Filter state, year, and crop
    mask = (
        df["state_fips"].isin(state_fips)
        & (df["year"] >= year_min)
        & (df["year"] <= year_max)
    )
    if crops is not None:
        mask &= df["crop"].isin(crops)
    df = df[mask].copy()

    # Deduplicate: max acres_harvested per fips/year/crop
    df = df.groupby(["fips", "year", "crop"], as_index=False)["acres_harvested"].max()

    # Drop nulls and zeros
    df = df.dropna(subset=["acres_harvested"])
    df = df[df["acres_harvested"] > 0]

    logger.info(
        f"Loaded NASS acreage: {len(df)} rows, "
        f"{df['fips'].nunique()} counties, "
        f"years {df['year'].min()}-{df['year'].max()}"
    )
    return df.drop(columns=["state_fips"], errors="ignore")


def load_climate(
    state_fips: list[str],
    year_min: int = 1950,
    year_max: int = 2025,
) -> pd.DataFrame:
    """Load annual climate data for target states.

    Args:
        state_fips: 2-digit state FIPS strings.
        year_min: Start year.
        year_max: End year.

    Returns:
        DataFrame with climate columns, one row per county-year.
    """
    df = pd.read_parquet(CLIMATE_PATH)
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["fips"].str[:2]

    mask = (
        df["state_fips"].isin(state_fips)
        & (df["year"] >= year_min)
        & (df["year"] <= year_max)
    )
    df = df[mask].copy()

    # Filter out state/division aggregates (FIPS ending in 998/999)
    df = df[~df["fips"].str[-3:].isin(["998", "999"])]

    logger.info(
        f"Loaded climate: {len(df)} rows, "
        f"{df['fips'].nunique()} counties, "
        f"years {df['year'].min()}-{df['year'].max()}"
    )
    return df.drop(columns=["state_fips"], errors="ignore")


def load_climate_monthly(
    state_fips: list[str],
    year_min: int = 1950,
    year_max: int = 2025,
    months: list[str] | None = None,
) -> pd.DataFrame:
    """Load monthly climate data for target states.

    Includes month-level tmax, tmin, tavg, precip, pdsi, and cdd columns,
    plus the annual summaries. Useful for capturing phenology-sensitive
    signals like spring temperature disruption for cotton.

    Args:
        state_fips: 2-digit state FIPS strings.
        year_min: Start year.
        year_max: End year.
        months: Optional list of month-var prefixes to keep (e.g.,
            ['tmin_m03', 'tmin_m04']). None means keep all.

    Returns:
        DataFrame with monthly climate columns, one row per county-year.
    """
    df = pd.read_parquet(CLIMATE_MONTHLY_PATH)
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["fips"].str[:2]

    mask = (
        df["state_fips"].isin(state_fips)
        & (df["year"] >= year_min)
        & (df["year"] <= year_max)
    )
    df = df[mask].copy()
    df = df[~df["fips"].str[-3:].isin(["998", "999"])]

    if months is not None:
        keep_cols = ["fips", "year"] + [c for c in df.columns if c in months]
        df = df[keep_cols]

    logger.info(
        f"Loaded monthly climate: {len(df)} rows, "
        f"{df['fips'].nunique()} counties, {df.shape[1]} columns"
    )
    return df.drop(columns=["state_fips"], errors="ignore")


def compute_acreage_shares(acreage: pd.DataFrame) -> pd.DataFrame:
    """Compute each crop's share of total county acreage per year.

    Args:
        acreage: DataFrame with [fips, year, crop, acres_harvested].

    Returns:
        Same DataFrame with an additional 'share' column (0-1).
    """
    total = acreage.groupby(["fips", "year"])["acres_harvested"].sum().reset_index()
    total.rename(columns={"acres_harvested": "total_acres"}, inplace=True)

    out = acreage.merge(total, on=["fips", "year"], how="left")
    out["share"] = out["acres_harvested"] / out["total_acres"]
    return out


def build_climate_features(
    climate: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Build climate trend features for each county.

    Computes rolling-window trends (slope) and level features for:
    - tmax_growing_avg, tmin_growing_avg, precip_growing_total
    - tmax_july, pdsi_growing_avg, cdd_annual

    Args:
        climate: Annual climate data.
        window: Rolling window size (years) for trend computation.

    Returns:
        DataFrame with original climate cols plus trend columns.
    """
    climate = climate.sort_values(["fips", "year"])
    out = climate.copy()

    trend_cols = [
        "tmax_growing_avg", "tmin_growing_avg", "precip_growing_total",
        "tmax_july", "pdsi_growing_avg", "cdd_annual",
    ]

    for col in trend_cols:
        if col not in out.columns:
            continue
        # Rolling mean (level)
        out[f"{col}_mean{window}y"] = (
            out.groupby("fips")[col]
            .transform(lambda s: s.rolling(window, min_periods=3).mean())
        )
        # Rolling trend (slope over window)
        out[f"{col}_trend{window}y"] = (
            out.groupby("fips")[col]
            .transform(
                lambda s: s.rolling(window, min_periods=3).apply(
                    _linear_slope, raw=True
                )
            )
        )

    return out


def _linear_slope(arr: np.ndarray) -> float:
    """Compute OLS slope of array against index (years).

    Args:
        arr: 1-D array of values.

    Returns:
        Slope (change per year).
    """
    n = len(arr)
    if n < 3:
        return np.nan
    x = np.arange(n, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() < 3:
        return np.nan
    x_m, y_m = x[mask], arr[mask]
    xbar = x_m.mean()
    slope = np.sum((x_m - xbar) * (y_m - y_m.mean())) / np.sum((x_m - xbar) ** 2)
    return slope


def _compute_period_change(
    acreage: pd.DataFrame,
    crop: str,
    pre_years: Tuple[int, int],
    post_years: Tuple[int, int],
) -> pd.DataFrame:
    """Compute change in acreage share between two periods for one crop.

    Args:
        acreage: Acreage data with 'share' column.
        crop: Target crop.
        pre_years: (start, end) of baseline period.
        post_years: (start, end) of event period.

    Returns:
        DataFrame with [fips, share_pre, share_post, share_change].
    """
    crop_data = acreage[acreage["crop"] == crop].copy()

    pre = crop_data[
        (crop_data["year"] >= pre_years[0]) & (crop_data["year"] <= pre_years[1])
    ]
    post = crop_data[
        (crop_data["year"] >= post_years[0]) & (crop_data["year"] <= post_years[1])
    ]

    pre_avg = pre.groupby("fips")["share"].mean().reset_index()
    pre_avg.rename(columns={"share": "share_pre"}, inplace=True)

    post_avg = post.groupby("fips")["share"].mean().reset_index()
    post_avg.rename(columns={"share": "share_post"}, inplace=True)

    merged = pre_avg.merge(post_avg, on="fips", how="outer").fillna(0)
    merged["share_change"] = merged["share_post"] - merged["share_pre"]

    return merged


# -----------------------------------------------------------------------
# Event 1: Sorghum expansion in southern Plains (1950-1975)
# -----------------------------------------------------------------------

def validate_sorghum_expansion() -> dict:
    """Test 1 (POSITIVE): Sorghum expansion in KS/OK/TX/NE 1950-1975.

    Train on 1950-1960, predict 1961-1975 sorghum acreage share changes.
    Pass criterion: Spearman rank correlation > 0.55 between predicted and
    actual county-level expansion.

    Returns:
        Dict with test results.
    """
    logger.info("=" * 60)
    logger.info("EVENT 1: Sorghum expansion, southern Plains 1950-1975")
    logger.info("=" * 60)

    states = ["20", "40", "48", "31"]  # KS, OK, TX, NE
    train_years = (1950, 1960)
    predict_years = (1961, 1975)

    # Load data
    acreage = load_nass_acreage(states, year_min=1950, year_max=1975)
    acreage = compute_acreage_shares(acreage)
    climate = load_climate(states, year_min=1950, year_max=1975)
    climate = build_climate_features(climate, window=5)

    # Compute actual sorghum share change
    actual_change = _compute_period_change(
        acreage, "sorghum", train_years, predict_years
    )
    logger.info(
        f"Actual sorghum change: {len(actual_change)} counties, "
        f"mean change = {actual_change['share_change'].mean():.3f}"
    )

    # Build training data: county-year panel with climate features
    # Target: sorghum acreage share
    sorghum_shares = acreage[acreage["crop"] == "sorghum"][
        ["fips", "year", "share"]
    ].copy()

    train_panel = sorghum_shares.merge(climate, on=["fips", "year"], how="inner")

    # Feature columns (all climate features)
    feature_cols = [
        c for c in train_panel.columns
        if c not in ("fips", "year", "share", "crop")
        and train_panel[c].dtype in ("float64", "float32", "int64")
    ]

    # Split
    train_mask = train_panel["year"].between(*train_years)
    predict_mask = train_panel["year"].between(*predict_years)

    X_train = train_panel.loc[train_mask, feature_cols].copy()
    y_train = train_panel.loc[train_mask, "share"].copy()
    X_predict = train_panel.loc[predict_mask, feature_cols].copy()
    fips_predict = train_panel.loc[predict_mask, "fips"].copy()

    # Handle NaNs
    X_train = X_train.fillna(X_train.median())
    X_predict = X_predict.fillna(X_predict.median())

    if len(X_train) < 20 or len(X_predict) < 20:
        logger.error(
            f"Insufficient data: train={len(X_train)}, predict={len(X_predict)}"
        )
        return {
            "event": "Sorghum expansion in southern Plains 1950-1975",
            "test_type": "POSITIVE",
            "criterion": "Spearman rank correlation > 0.55",
            "passed": False,
            "reason": "Insufficient data",
        }

    logger.info(f"Training on {len(X_train)} obs, predicting {len(X_predict)} obs")
    logger.info(f"Features: {len(feature_cols)}")

    # Train LightGBM regressor to predict sorghum share from climate
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Predict on event-period data
    pred_share = model.predict(X_predict)

    # Average predictions per county
    pred_df = pd.DataFrame({
        "fips": fips_predict.values,
        "pred_share": pred_share,
    })
    pred_avg = pred_df.groupby("fips")["pred_share"].mean().reset_index()

    # Also compute the actual average share during training period per county
    train_share_avg = (
        sorghum_shares[sorghum_shares["year"].between(*train_years)]
        .groupby("fips")["share"]
        .mean()
        .reset_index()
    )
    train_share_avg.rename(columns={"share": "train_share"}, inplace=True)

    # Predicted change = predicted share in event period - actual share in training
    pred_avg = pred_avg.merge(train_share_avg, on="fips", how="left")
    pred_avg["train_share"] = pred_avg["train_share"].fillna(0)
    pred_avg["pred_change"] = pred_avg["pred_share"] - pred_avg["train_share"]

    # Merge with actual change
    comparison = pred_avg.merge(actual_change, on="fips", how="inner")
    comparison = comparison.dropna(subset=["pred_change", "share_change"])

    if len(comparison) < 10:
        logger.error(f"Too few counties in comparison: {len(comparison)}")
        return {
            "event": "Sorghum expansion in southern Plains 1950-1975",
            "test_type": "POSITIVE",
            "criterion": "Spearman rank correlation > 0.55",
            "passed": False,
