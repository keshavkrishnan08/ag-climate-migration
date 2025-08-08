"""Robustness checks for Nature Food reviewers.

Six checks addressing the top reviewer concerns:
  1. Hedonic with soil quality proxy (historical yield 1990-2005 as soil proxy)
  2. Leave-one-crop-out sensitivity for stranded asset computation
  3. Leave-one-GCM-out sensitivity for stranded asset computation
  4. Placebo test: run cascade on LEAST climate-affected counties (top quartile positive impact)
  5. Temporal stability of the hedonic regression (2010-2015 vs 2015-2022)
  6. Insurance mispricing under alternative coverage levels (65% and 85%)

Each check saves results to results/robustness/ and prints a one-line verdict:
  ROBUST   — result is insensitive to the specification change
  SENSITIVE — result changes materially; report both
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from loguru import logger
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROJ    = PROJECT_ROOT / "data" / "projections"
RESULTS_DIR  = PROJECT_ROOT / "results" / "robustness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

CPI_2023 = 304.7
CPI_2022 = 296.8
DEFLATOR_2022 = CPI_2023 / CPI_2022
SEED = 42
np.random.seed(SEED)

GROWING_MONTHS = [4, 5, 6, 7, 8, 9]

COMMODITY_PRICES = {
    "corn": 5.50, "soybeans": 12.80, "wheat_winter": 7.20,
    "wheat_spring": 8.10, "cotton": 0.78, "sorghum": 5.30,
    "barley": 6.10, "oats": 3.80,
}

# From the main hedonic run — reference coefficients for stability comparison
BASELINE_BETA_TMAX     = None   # filled in at runtime from check 1 baseline run
BASELINE_STRANDED_B    = 168.0  # hedonic $168B from headline_numbers_preliminary.json
DCF_CONSERVATIVE_B     = 76.0   # DCF conservative from same file

VERDICTS: list[dict] = []


# ──────────────────────────────────────────────────────────────────────────────
# Shared data loader (load once, pass around)
# ──────────────────────────────────────────────────────────────────────────────

def load_shared_data() -> dict:
    """Load all datasets needed by robustness checks.

    Returns:
        Dict with keys: land_values, climate_monthly, acs, nass_yields,
        climate_proj, yield_proj, rma_data, tipping_df.
    """
    logger.info("Loading shared datasets…")

    land_values = pd.read_parquet(
        DATA_RAW / "nass" / "nass_land_values.parquet",
        columns=["fips", "year", "land_value_per_acre"],
    )
    climate_monthly = pd.read_parquet(
        DATA_RAW / "prism" / "county_climate_monthly.parquet",
        columns=(["fips", "year", "tmax_m07"] +
                 [f"precip_m{m:02d}" for m in GROWING_MONTHS]),
    )
    acs = pd.read_parquet(
        DATA_RAW / "census" / "acs_county_demographics.parquet",
        columns=["fips", "year", "total_population", "median_household_income"],
    )
    nass_yields = pd.read_parquet(
        DATA_RAW / "nass" / "nass_county_yields.parquet",
        columns=["fips", "year", "crop", "yield_bu_acre", "acres_harvested"],
    )
    climate_proj = pd.read_parquet(
        DATA_PROJ / "county_climate_projections.parquet",
        columns=["fips", "year", "scenario", "delta_tmax_july", "delta_precip_growing"],
    )
    climate_proj = climate_proj[climate_proj["scenario"] == "SSP245"].copy()

    yield_proj = pd.read_parquet(
        DATA_PROJ / "yield_projections_SSP245.parquet",
        columns=["fips", "year", "crop", "scenario",
                 "yield_projected", "yield_baseline",
                 "climate_impact_bu", "acres_harvested"],
    )

    rma_path = DATA_RAW / "rma" / "rma_sob_all_years.parquet"
    rma_data = pd.read_parquet(
        rma_path,
        columns=["year", "fips", "crop_name", "acres", "total_premium",
                 "indemnity", "premium_per_acre"],
    ) if rma_path.exists() else pd.DataFrame()

    tipping_df = pd.read_parquet(
        PROJECT_ROOT / "results" / "cascade" / "tipping_points_SSP245.parquet"
    )

    logger.info(
        f"  land_values={len(land_values)}, climate_monthly={len(climate_monthly)}, "
        f"nass_yields={len(nass_yields)}, yield_proj={len(yield_proj)}, "
        f"tipping_df={len(tipping_df)}"
    )
    return dict(
        land_values=land_values,
        climate_monthly=climate_monthly,
        acs=acs,
        nass_yields=nass_yields,
        climate_proj=climate_proj,
        yield_proj=yield_proj,
        rma_data=rma_data,
        tipping_df=tipping_df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Hedonic helper — reused across checks 1 and 5
# ──────────────────────────────────────────────────────────────────────────────

def _build_cross_section(
    land_values: pd.DataFrame,
    climate_monthly: pd.DataFrame,
    acs: pd.DataFrame,
    nass_yields: pd.DataFrame,
    lv_years: list[int],
    clim_years: tuple[int, int],
    acs_years: tuple[int, int],
) -> pd.DataFrame:
    """Build hedonic cross-section for a given temporal window.

    Args:
        land_values: NASS land value panel.
        climate_monthly: PRISM monthly climate panel.
        acs: ACS demographics panel.
        nass_yields: NASS county yields (for farm-acres calibration).
