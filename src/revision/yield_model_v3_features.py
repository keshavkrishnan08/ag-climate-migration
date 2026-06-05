"""Revision: yield model with modern agro-climatic features (Reviewer 2 #1).

Reviewer 2 argued the yield model is under-specified relative to current
practice: it omits vapour pressure deficit (VPD), explicit heat-stress /
extreme-degree-day exposure, and soil-moisture stress -- all standard in the
modern crop-yield literature. This script engineers those features from the
monthly nClimDiv panel (tmax/tmin/precip/PDSI) and retrains the same three-model
ensemble (LightGBM + Ridge + RandomForest, NNLS blend) on the SAME temporal
split (train <= 2012, held-out test 2013-2023), so the improvement is
attributable to the features, not to leakage.

New features (growing season = months 04-09):
  vpd_growing      : mean vapour pressure deficit (kPa), FAO-56 with tmin as the
                     dewpoint proxy (ea = es(tmin), VPD = es(tmax) - es(tmin)).
  vpd_july         : July VPD (peak-demand month).
  edd30_growing    : extreme degree-days above 30 C, approximated month-by-month
                     from monthly tmax via a within-month sinusoid.
  heat_days_proxy  : count of growing-season months with tmax > 30 C.
  sm_stress        : soil-moisture stress = -min(growing-season PDSI) (driest
                     month deficit; larger = drier).
  sm_stress_july   : -PDSI in July.
  vpd_x_sm         : VPD x soil-moisture stress interaction (compound heat-drought).

All anomaly-standardised against the county mean so the model predicts the
z-scored yield anomaly (the existing target). Seed 42.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "revision"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
GROW_MONTHS = [f"{m:02d}" for m in range(4, 10)]   # Apr-Sep


def es_kpa(t_c):
    """Saturation vapour pressure (kPa) from temperature in C (FAO-56 Tetens)."""
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def build_modern_features():
    """Engineer VPD, EDD>30C, heat-day, and soil-moisture stress features.

    Returns:
        DataFrame [fips, year, <new feature columns>].
    """
    m = pd.read_parquet(DATA_RAW / "prism" / "county_climate_monthly.parquet")
    m["fips"] = m["fips"].astype(str).str.zfill(5)

    # Convert F -> C for grow-season months
    for mm in GROW_MONTHS:
        m[f"tmaxc_{mm}"] = (m[f"tmax_m{mm}"] - 32) * 5 / 9
        m[f"tminc_{mm}"] = (m[f"tmin_m{mm}"] - 32) * 5 / 9

    # VPD per month: es(tmax) - es(tmin); growing-season + July means
    vpd_cols = []
    for mm in GROW_MONTHS:
        m[f"vpd_{mm}"] = (es_kpa(m[f"tmaxc_{mm}"]) - es_kpa(m[f"tminc_{mm}"])).clip(lower=0)
        vpd_cols.append(f"vpd_{mm}")
    m["vpd_growing"] = m[vpd_cols].mean(axis=1)
    m["vpd_july"] = m["vpd_07"]

    # EDD above 30 C, month-by-month sinusoid approx between tmin and tmax.
    # For a within-month diurnal sinusoid with min=tmin,max=tmax, the mean daily
    # degree-days above threshold T0 has a closed form; we approximate with a
    # 24-point quadrature per month and scale by ~30 days.
    thr = 30.0
    edd = np.zeros(len(m))
    hot = np.zeros(len(m))
    phase = np.linspace(0, np.pi, 24)
    for mm in GROW_MONTHS:
        tmn = m[f"tminc_{mm}"].values
        tmx = m[f"tmaxc_{mm}"].values
        mid = (tmx + tmn) / 2.0
