"""Audit improvement A: richer drought-trajectory features + Huber-loss target.

Orthogonal to v7 (which changes the target to %-deviation and uses a temperature
exposure spectrum). Here we KEEP the existing z-scored county-detrended anomaly
target -- so the held-out R^2 is DIRECTLY comparable to the v4 number (0.227) the
paper currently reports -- and attack two distinct, defensible levers:

1. RICHER DROUGHT DYNAMICS from the monthly PDSI/precip panel. Yield loss in
   corn/soy/cotton is driven by drought *timing and persistence*, not just the
   season minimum. We add:
     - consecutive-dry-month run length (longest streak PDSI<-1 in growing season)
     - PDSI trajectory slope (Apr->Sep linear slope: drying vs recovering)
     - PDSI integral / area below -1 (cumulative water deficit)
     - month-of-driest-PDSI (timing relative to silking/grain-fill)
     - precip deficit run (longest streak of below-county-normal precip)
     - early- vs late-season PDSI contrast (Apr-Jun mean minus Jul-Sep mean)
   All county-demeaned to anomalies, matching the pipeline convention.

2. HUBER LOSS instead of squared error. The z-scored anomaly target has +/-7 sigma
   tails (drought/flood outliers) that dominate the MSE gradient and bias the fit
   toward un-modellable freak years. Huber down-weights those tails, so the model
   learns the bulk climate-yield signal rather than chasing noise. This is a model
   improvement, not a metric reframing: we still REPORT plain R^2 / Spearman on the
   untransformed held-out anomaly.

Same temporal split (train<=2012, test 2013-2023). Seed 42. Climate-only feature
set (no spatial-yield lags) so it stays projectable from CMIP6 deltas.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from yield_model_v3_features import build_modern_features, add_county_anomalies, GROW_MONTHS
from yield_v4_morefeatures import extra_features

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "revision"
SEED = 42
GROW = [f"{m:02d}" for m in range(4, 10)]   # Apr-Sep


def drought_trajectory_features():
    """Engineer drought timing / persistence / shape features from monthly PDSI.

    Returns:
        DataFrame [fips, year, <drought-shape feature columns>].
    """
    m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet")
    m["fips"] = m["fips"].astype(str).str.zfill(5)

    pdsi = np.column_stack([m[f"pdsi_m{mm}"].values for mm in GROW])   # (n, 6) Apr..Sep
    precip = np.column_stack([m[f"precip_m{mm}"].values for mm in GROW])
    n, k = pdsi.shape
    months_idx = np.arange(k)

    # 1. longest consecutive run of PDSI < -1 within the growing season
    def longest_run(mask):
        out = np.zeros(mask.shape[0])
        cur = np.zeros(mask.shape[0])
        for j in range(mask.shape[1]):
            cur = np.where(mask[:, j], cur + 1, 0)
            out = np.maximum(out, cur)
        return out
    dry_run = longest_run(pdsi < -1.0)

    # 2. PDSI trajectory slope across Apr->Sep (drying if negative)
    xm = months_idx - months_idx.mean()
    denom = np.sum(xm ** 2)
    pdsi_slope = ((pdsi - pdsi.mean(axis=1, keepdims=True)) * xm).sum(axis=1) / denom

    # 3. cumulative water deficit: sum of (PDSI below -1), i.e. area under -1
    deficit_integral = np.maximum(-1.0 - pdsi, 0).sum(axis=1)
