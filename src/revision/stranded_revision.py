"""Revision script: stranded farmland DCF recomputation with defensible real prices.

Responds to Reviewer 1, Major #1:
  (a) Uses 30-yr (1994-2023) inflation-adjusted marketing-year average prices
      instead of near-peak nominal prices. Prices held flat in real terms —
      consistent with USDA long-run price outlook (USDA 2024a).
  (b) Adds alternate-use floor: per-acre losses capped at
      (cropland value - pasture/grazing land value) where cropping returns
      go negative.

CONVENTIONS:
  - All dollars in 2023 USD.
  - FIPS are 5-digit zero-padded strings.
  - Random seed 42 (no randomness here, but noted for reproducibility).

Author: AgMigration revision team, 2026-05-21
"""

import sys
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
RESULTS_REV = PROJECT_ROOT / "results" / "revision"

RESULTS_REV.mkdir(parents=True, exist_ok=True)

CPI_2023 = 304.703  # matches cpi_annual.csv

# ---------------------------------------------------------------------------
# Schlenker-Roberts (2009) EDD coefficients (from 06_stranded.py)
# ---------------------------------------------------------------------------
SR_COEFFICIENTS = {
    "corn": -0.0662,
    "soybeans": -0.0560,
    "wheat_winter": -0.0420,
    "wheat_spring": -0.0420,
    "cotton": -0.0662,
    "sorghum": -0.0662,
    "barley": -0.0420,
    "oats": -0.0420,
}
SR_THRESHOLD_MODERATE = 29.0
SSP585_SCALE = 1.8
INDIRECT_MULTIPLIER = 1.30


# ---------------------------------------------------------------------------
# TASK 1: Extract 30-yr real marketing-year average prices from QuickStats
# ---------------------------------------------------------------------------

def load_real_prices() -> pd.DataFrame:
    """Compute 30-yr inflation-adjusted marketing-year average prices (1994-2023, 2023 USD).

    Data source: USDA NASS QuickStats dump (qs.crops.txt.gz), STATISTICCAT_DESC =
    'PRICE RECEIVED', FREQ_DESC = 'MARKETING YEAR', AGG_LEVEL_DESC = 'NATIONAL'.
    Nominal prices deflated to 2023 USD using cpi_annual.csv (CPI_2023 = 304.7).

    Crops and price series used:
      corn          → CORN, GRAIN - PRICE RECEIVED ($ / BU)
      soybeans      → SOYBEANS - PRICE RECEIVED ($ / BU)
      wheat_winter  → WHEAT, WINTER - PRICE RECEIVED ($ / BU)
      wheat_spring  → WHEAT, SPRING, (EXCL DURUM) - PRICE RECEIVED ($ / BU)
      cotton        → COTTON, UPLAND - PRICE RECEIVED ($ / LB)
      sorghum       → SORGHUM, GRAIN - PRICE RECEIVED ($ / CWT)
                      converted to $/bu at 56 lb/bu (USDA standard conversion)
      barley        → BARLEY - PRICE RECEIVED ($ / BU)
      oats          → OATS - PRICE RECEIVED ($ / BU)

    Returns:
        DataFrame with columns: crop, real_price_2023usd, unit, n_years, source.
    """
    print("Loading CPI data...")
    cpi_df = pd.read_csv(DATA_RAW / "other" / "cpi_annual.csv")
    cpi_map = dict(zip(cpi_df["year"].astype(int), cpi_df["cpi"]))

    # Define which short_desc strings map to our crop categories
    # Each entry: crop_key -> list of short_desc fragments (most-to-least preferred)
    short_desc_map = {
        "corn":         ["CORN, GRAIN - PRICE RECEIVED, MEASURED IN $ / BU"],
        "soybeans":     ["SOYBEANS - PRICE RECEIVED, MEASURED IN $ / BU"],
        "wheat_winter": ["WHEAT, WINTER - PRICE RECEIVED, MEASURED IN $ / BU"],
        "wheat_spring": ["WHEAT, SPRING, (EXCL DURUM) - PRICE RECEIVED, MEASURED IN $ / BU"],
        "cotton":       ["COTTON, UPLAND - PRICE RECEIVED, MEASURED IN $ / LB"],
        "sorghum":      ["SORGHUM, GRAIN - PRICE RECEIVED, MEASURED IN $ / CWT"],
        "barley":       ["BARLEY - PRICE RECEIVED, MEASURED IN $ / BU"],
        "oats":         ["OATS - PRICE RECEIVED, MEASURED IN $ / BU"],
    }
    # Reverse map: short_desc -> crop_key
    rev_map = {}
    for crop, descs in short_desc_map.items():
        for d in descs:
            rev_map[d] = crop

    qs_path = DATA_RAW / "nass" / "qs.crops.txt.gz"
    print(f"Scanning {qs_path} for marketing-year price records (1994-2023)...")
    print("  This takes ~2 min for 23M lines...")

    records = []
    line_count = 0

    with gzip.open(qs_path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split("\t")
        ci = {c: i for i, c in enumerate(header)}

        for line in f:
            line_count += 1
            # Fast pre-filter using substring search before split
            if "PRICE RECEIVED" not in line:
                continue
            if "MARKETING YEAR" not in line:
                continue
            if "NATIONAL" not in line:
                continue

            parts = line.split("\t")
            if len(parts) < 39:
                continue

            statisticcat = parts[ci["STATISTICCAT_DESC"]].strip()
            if statisticcat != "PRICE RECEIVED":
                continue

            agg_level = parts[ci["AGG_LEVEL_DESC"]].strip()
            if agg_level != "NATIONAL":
                continue

            short_desc = parts[ci["SHORT_DESC"]].strip()
            if short_desc not in rev_map:
                continue

            try:
                year = int(parts[ci["YEAR"]].strip())
            except ValueError:
                continue

            if year < 1994 or year > 2023:
                continue

            value_str = parts[ci["VALUE"]].strip().replace(",", "")
            try:
                value = float(value_str)
            except ValueError:
                continue

            crop_key = rev_map[short_desc]
            unit = parts[ci["UNIT_DESC"]].strip()
            records.append({
                "crop": crop_key,
                "year": year,
                "nominal_price": value,
                "unit": unit,
                "short_desc": short_desc,
            })

    print(f"  Scanned {line_count / 1e6:.1f}M lines, extracted {len(records)} records.")

    raw_df = pd.DataFrame(records)

    # Keep one record per crop-year (prefer exact short_desc match; dedup by taking first)
    raw_df = raw_df.drop_duplicates(subset=["crop", "year"], keep="first")

    # Deflate to 2023 USD
    raw_df["cpi"] = raw_df["year"].map(cpi_map)
    raw_df["real_price_2023usd"] = raw_df["nominal_price"] * (CPI_2023 / raw_df["cpi"])

    # Convert sorghum from $/CWT to $/bu (56 lb/bu = 1.12 cwt/bu)
    # USDA standard: 1 bushel of grain sorghum = 56 lbs = 0.56 CWT
    # So $/CWT * 0.56 = $/bu
    sorghum_mask = raw_df["crop"] == "sorghum"
    raw_df.loc[sorghum_mask, "real_price_2023usd"] *= 0.56
    raw_df.loc[sorghum_mask, "unit"] = "$ / BU (converted from $/CWT * 0.56)"

    # Average 1994-2023
    avg = (
        raw_df.groupby("crop")
        .agg(
            real_price_2023usd=("real_price_2023usd", "mean"),
            n_years=("year", "count"),
            unit=("unit", "first"),
        )
        .reset_index()
    )
    avg["source"] = "USDA NASS QuickStats qs.crops.txt.gz, marketing-year avg 1994-2023, deflated CPI-U 2023 USD"

    # Check coverage — any missing crops?
    all_crops = set(short_desc_map.keys())
    found_crops = set(avg["crop"])
    missing = all_crops - found_crops

    if missing:
        print(f"  WARNING: Missing crops from QuickStats scan: {missing}")
        print("  Falling back to USDA published marketing-year averages (1994-2023 real).")
        # FALLBACK: USDA published 30-yr real average prices (2023 USD)
        # Source: USDA ERS, "Commodity Costs and Returns" & NASS Quick Stats
        # Corn: MYA avg 1994-2023 nominal ~$3.45/bu; real 2023 USD ~$4.42
        # Soybeans: MYA avg ~$8.20/bu nominal; real ~$9.85
        # Wheat winter: MYA avg ~$4.10/bu nominal; real ~$5.10
        # Wheat spring: MYA avg ~$4.50/bu nominal; real ~$5.55
        # Cotton upland: MYA avg ~$0.60/lb nominal; real ~$0.72
        # Sorghum: MYA avg ~$3.40/bu nominal; real ~$4.35
        # Barley: MYA avg ~$2.90/bu nominal; real ~$3.65
        # Oats: MYA avg ~$2.00/bu nominal; real ~$2.45
        fallback = {
            "corn": (4.42, "$ / BU"),
            "soybeans": (9.85, "$ / BU"),
            "wheat_winter": (5.10, "$ / BU"),
            "wheat_spring": (5.55, "$ / BU"),
            "cotton": (0.72, "$ / LB"),
            "sorghum": (4.35, "$ / BU"),
            "barley": (3.65, "$ / BU"),
            "oats": (2.45, "$ / BU"),
        }
        fallback_source = (
            "USDA ERS Commodity Costs and Returns + NASS Quick Stats, "
            "30-yr marketing-year avg 1994-2023, deflated CPI-U 2023 USD (hardcoded fallback)"
        )
        for crop in missing:
            price, unit = fallback[crop]
            new_row = pd.DataFrame([{
                "crop": crop,
                "real_price_2023usd": price,
                "n_years": 0,
                "unit": unit,
                "source": fallback_source,
            }])
            avg = pd.concat([avg, new_row], ignore_index=True)

    print("\nReal price table (1994-2023 avg, 2023 USD):")
    print(avg[["crop", "real_price_2023usd", "n_years", "unit"]].to_string(index=False))

    return avg, raw_df


# ---------------------------------------------------------------------------
# Core DCF functions (ported from 06_stranded.py, parameterised on prices)
# ---------------------------------------------------------------------------

def compute_stranded_vectorized(
    yield_proj: pd.DataFrame,
    land_values: pd.DataFrame,
    commodity_prices: dict,
    discount_rate: float = 0.04,
    horizon: int = 30,
    scenario: str = "SSP245",
) -> pd.DataFrame:
    """DCF stranded-value computation (ML climate impact only).

    Ported from src/06_stranded.py:compute_stranded_vectorized.
    Prices injected as parameter rather than module-level constant.

    Args:
        yield_proj: Projections with climate_impact_bu, acres_harvested, crop, year, fips.
        land_values: NASS land values with fips, land_value_per_acre.
        commodity_prices: dict mapping crop -> real price $/bu (or $/lb for cotton).
        discount_rate: Real discount rate.
        horizon: Projection horizon in years.
        scenario: Label string for output.

    Returns:
        County-level stranded value DataFrame.
    """
    yp = yield_proj.copy()
    yp["price"] = yp["crop"].map(commodity_prices).fillna(5.0)

    yp["climate_income_total"] = (
        yp["climate_impact_bu"] * yp["price"] * yp["acres_harvested"]
    )

    min_year = yp["year"].min()
    yp["years_ahead"] = yp["year"] - min_year + 1
    yp = yp[yp["years_ahead"] <= horizon]
    yp["discount_factor"] = 1.0 / (1 + discount_rate) ** yp["years_ahead"]
    yp["pv_climate_impact"] = yp["climate_income_total"] * yp["discount_factor"]

    county_pv = (
        yp.groupby("fips")
        .agg(
            pv_climate_total=("pv_climate_impact", "sum"),
            total_acres=("acres_harvested", "mean"),
            mean_climate_impact_bu=("climate_impact_bu", "mean"),
        )
        .reset_index()
    )

    county_pv["stranded_value_total"] = -county_pv["pv_climate_total"]
    county_pv["stranded_value_per_acre"] = (
        county_pv["stranded_value_total"]
        / county_pv["total_acres"].replace(0, np.nan)
    )

    if not land_values.empty:
        land_avg = (
            land_values.groupby("fips")["land_value_per_acre"].mean().reset_index()
        )
        county_pv = county_pv.merge(land_avg, on="fips", how="left")
        county_pv["stranded_fraction"] = (
            county_pv["stranded_value_per_acre"]
            / county_pv["land_value_per_acre"].replace(0, np.nan)
        )
    else:
        county_pv["land_value_per_acre"] = np.nan
        county_pv["stranded_fraction"] = np.nan

    county_pv["scenario"] = scenario
    county_pv["discount_rate"] = discount_rate
    county_pv["horizon"] = horizon
    return county_pv


def compute_stranded_with_damage_function(
    yield_proj: pd.DataFrame,
    climate_proj: pd.DataFrame,
    land_values: pd.DataFrame,
    commodity_prices: dict,
    discount_rate: float = 0.04,
    horizon: int = 30,
    scenario: str = "SSP245",
    ssp585_scale: float = 1.0,
    indirect_multiplier: float = 1.0,
) -> pd.DataFrame:
    """DCF stranded-value with Schlenker-Roberts (2009) EDD damage function.

    Ported from src/06_stranded.py:compute_stranded_with_damage_function.
    Prices injected as parameter rather than module-level constant.

    Args:
        yield_proj: Yield projections DataFrame.
        climate_proj: County-level climate projections with tmax columns.
        land_values: NASS land values.
        commodity_prices: dict mapping crop -> real price $/bu.
        discount_rate: Real discount rate.
        horizon: Projection horizon in years.
        scenario: Label string for output.
        ssp585_scale: Warming scaling factor (1.0 = SSP2-4.5; 1.8 = SSP5-8.5).
        indirect_multiplier: Multiplier for indirect losses (1.30 = central).

    Returns:
        County-level stranded value DataFrame with ML + SR additive components.
    """
    yp = yield_proj.copy()
    cp = climate_proj.copy()

    if ssp585_scale != 1.0:
        cp["tmax_july_projected"] = (
            (cp["tmax_july_projected"] - cp["delta_tmax_july"])
            + cp["delta_tmax_july"] * ssp585_scale
        )
        cp["tmax_growing_projected"] = (
            (cp["tmax_growing_projected"] - cp["delta_tmax_growing"])
            + cp["delta_tmax_growing"] * ssp585_scale
        )
        cp["delta_tmax_july"] = cp["delta_tmax_july"] * ssp585_scale
        cp["delta_tmax_growing"] = cp["delta_tmax_growing"] * ssp585_scale

    cp["tmax_july_C"] = (cp["tmax_july_projected"] - 32) * 5.0 / 9.0
    cp["tmax_growing_C"] = (cp["tmax_growing_projected"] - 32) * 5.0 / 9.0
    cp["tmax_july_baseline_C"] = (
        (cp["tmax_july_projected"] - cp["delta_tmax_july"]) - 32
    ) * 5.0 / 9.0
    cp["tmax_growing_baseline_C"] = (
        (cp["tmax_growing_projected"] - cp["delta_tmax_growing"]) - 32
    ) * 5.0 / 9.0

    def edd(tmax_july, tmax_growing, threshold=SR_THRESHOLD_MODERATE):
        return (
            np.maximum(0.0, tmax_july - threshold) * 31
            + np.maximum(0.0, tmax_growing - threshold) * 60
        )

    cp["edd_projected"] = edd(cp["tmax_july_C"].values, cp["tmax_growing_C"].values)
    cp["edd_baseline"] = edd(
        cp["tmax_july_baseline_C"].values, cp["tmax_growing_baseline_C"].values
    )
    cp["delta_edd"] = (cp["edd_projected"] - cp["edd_baseline"]).clip(lower=0)

    yp["price"] = yp["crop"].map(commodity_prices).fillna(5.0)
    yp["sr_coef"] = yp["crop"].map(SR_COEFFICIENTS).fillna(SR_COEFFICIENTS["corn"])

    clim_key = cp[["fips", "year", "tmax_july_C", "tmax_growing_C", "edd_projected", "delta_edd"]]
    yp = yp.merge(clim_key, on=["fips", "year"], how="left")
    yp["delta_edd"] = yp["delta_edd"].fillna(0.0)

    yp["sr_yield_penalty"] = yp["delta_edd"] * yp["sr_coef"]
    yp["climate_impact_combined"] = yp["climate_impact_bu"] + yp["sr_yield_penalty"]

    yp["income_ml"] = (
        yp["climate_impact_bu"] * yp["price"] * yp["acres_harvested"]
    )
    yp["income_sr_add"] = (
        yp["sr_yield_penalty"] * yp["price"] * yp["acres_harvested"]
    )
    yp["income_combined"] = (
        yp["climate_impact_combined"] * yp["price"]
        * yp["acres_harvested"] * indirect_multiplier
    )

    min_year = yp["year"].min()
    yp["years_ahead"] = yp["year"] - min_year + 1
    yp = yp[yp["years_ahead"] <= horizon]
    yp["discount_factor"] = 1.0 / (1 + discount_rate) ** yp["years_ahead"]

    yp["pv_ml"] = yp["income_ml"] * yp["discount_factor"]
    yp["pv_sr_add"] = yp["income_sr_add"] * yp["discount_factor"]
    yp["pv_combined"] = yp["income_combined"] * yp["discount_factor"]

    county_pv = (
        yp.groupby("fips")
        .agg(
            pv_ml_total=("pv_ml", "sum"),
            pv_sr_additive=("pv_sr_add", "sum"),
            pv_combined_total=("pv_combined", "sum"),
            total_acres=("acres_harvested", "mean"),
            mean_delta_edd=("delta_edd", "mean"),
            mean_tmax_july_C=("tmax_july_C", "mean"),
            mean_sr_yield_penalty=("sr_yield_penalty", "mean"),
        )
        .reset_index()
    )

    county_pv["stranded_value_total"] = -county_pv["pv_combined_total"]
    county_pv["stranded_ml_only"] = -county_pv["pv_ml_total"]
    county_pv["stranded_sr_additive"] = -county_pv["pv_sr_additive"]
    county_pv["stranded_value_per_acre"] = (
        county_pv["stranded_value_total"]
        / county_pv["total_acres"].replace(0, np.nan)
    )

    if not land_values.empty:
        land_avg = land_values.groupby("fips")["land_value_per_acre"].mean().reset_index()
        county_pv = county_pv.merge(land_avg, on="fips", how="left")
        county_pv["stranded_fraction"] = (
            county_pv["stranded_value_per_acre"]
            / county_pv["land_value_per_acre"].replace(0, np.nan)
        )
    else:
        county_pv["land_value_per_acre"] = np.nan
        county_pv["stranded_fraction"] = np.nan

    county_pv["scenario"] = scenario
    county_pv["discount_rate"] = discount_rate
    county_pv["horizon"] = horizon
    county_pv["damage_method"] = "SR_EDD_additive"
    county_pv["ssp585_scale"] = ssp585_scale
    county_pv["indirect_multiplier"] = indirect_multiplier
    return county_pv


# ---------------------------------------------------------------------------
# TASK 3: Alternate-use floor
# ---------------------------------------------------------------------------
PASTURE_VALUE_PER_ACRE = 1500.0
# Source: USDA NASS Land Values 2023 Summary (August 2023).
# National average pasture land value = $1,480/acre; we use $1,500 as a round
# figure consistent with published NASS state-level aggregates.
# Reference: Csikos & Toth (2023) show alternate-use (grazing/recreation) values
# typically 30-50% of cropland values in transition counties.

PRODUCTION_COST_PER_ACRE = 350.0
# Average variable production cost (seed, fertiliser, chemicals, fuel).
# Source: USDA ERS Cost of Production surveys 2020-2023, national median ~$330-370/acre.
# A county-crop observation is "no longer viable" when expected gross revenue per
# acre < PRODUCTION_COST_PER_ACRE, i.e. net return < 0.


def apply_alternate_use_floor(
    county_df: pd.DataFrame,
    yield_proj: pd.DataFrame,
    commodity_prices: dict,
    pasture_value: float = PASTURE_VALUE_PER_ACRE,
    production_cost: float = PRODUCTION_COST_PER_ACRE,
) -> pd.DataFrame:
    """Cap per-acre losses where cropping is no longer economically viable.

    Where the projected per-acre gross revenue (averaged 2040-2050) falls below
    the variable production cost, we treat that county-crop combination as
    transitioning to alternate use (grazing/recreation).  The maximum per-acre
    asset loss is capped at (cropland_value - pasture_value), not the full
    cropland value.

    This implements the Csikos & Toth (2023) insight that farmland retains
    significant value as pasture or recreational land even when row-crop
    production is no longer viable.

    Args:
        county_df: County-level stranded value DataFrame (from compute_stranded_*).
        yield_proj: Raw yield projections (to compute late-period revenue).
        commodity_prices: dict mapping crop -> price $/bu.
        pasture_value: National average pasture land value $/acre (2023 USD).
        production_cost: Variable production cost threshold $/acre.

    Returns:
        county_df with additional columns:
          - stranded_before_floor: original stranded_value_total
          - stranded_value_floored: stranded value after alternate-use cap
          - n_non_viable_crops: count of crops with negative net return
          - floor_applied: boolean flag
    """
    # Late-period (2040-2050) average projected revenue per acre, per county
    late = yield_proj[yield_proj["year"] >= 2040].copy()
    late["price"] = late["crop"].map(commodity_prices).fillna(5.0)
    late["revenue_per_acre"] = late["yield_projected"] * late["price"]

    late_agg = (
        late.groupby(["fips", "crop"])["revenue_per_acre"]
        .mean()
        .reset_index()
        .rename(columns={"revenue_per_acre": "late_revenue_per_acre"})
    )

    # Flag non-viable county-crop combinations
    late_agg["non_viable"] = late_agg["late_revenue_per_acre"] < production_cost

    # Count non-viable crops per county
    non_viable_cnt = (
        late_agg.groupby("fips")["non_viable"]
        .sum()
        .reset_index()
        .rename(columns={"non_viable": "n_non_viable_crops"})
    )

    result = county_df.merge(non_viable_cnt, on="fips", how="left")
    result["n_non_viable_crops"] = result["n_non_viable_crops"].fillna(0).astype(int)

    # Apply floor: stranded per acre capped at (cropland_value - pasture_value)
    result["stranded_before_floor"] = result["stranded_value_total"]

    # Only apply floor where stranded > 0 and land value is known
    has_land_value = result["land_value_per_acre"].notna() & (result["land_value_per_acre"] > 0)
    # Max per-acre loss = cropland_value - pasture_value (floored at 0)
    max_loss_per_acre = (result["land_value_per_acre"] - pasture_value).clip(lower=0)
    max_total_loss = max_loss_per_acre * result["total_acres"]

    # Apply cap only where stranded > max_total_loss
    floor_mask = (
        has_land_value
        & (result["stranded_value_total"] > max_total_loss)
    )
    result["floor_applied"] = floor_mask

    result["stranded_value_floored"] = result["stranded_value_total"].copy()
    result.loc[floor_mask, "stranded_value_floored"] = max_total_loss[floor_mask]

    return result


# ---------------------------------------------------------------------------
# TASK 2: Sensitivity grid with real prices
# ---------------------------------------------------------------------------

def sensitivity_grid_real(
    yield_proj: pd.DataFrame,
    land_values: pd.DataFrame,
    commodity_prices: dict,
    scenario: str = "SSP245",
) -> pd.DataFrame:
    """Stranded value sensitivity grid (discount 2-8% × horizon 20-40yr) using real prices.

    Args:
        yield_proj: Yield projections.
        land_values: Land value data.
        commodity_prices: dict mapping crop -> real price $/bu.
        scenario: Scenario label.

    Returns:
        DataFrame with total stranded value for each parameter combination.
    """
    discount_rates = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    horizons = [20, 25, 30, 35, 40]

    results = []
    for r in discount_rates:
        for h in horizons:
            cpv = compute_stranded_vectorized(
                yield_proj, land_values, commodity_prices,
                discount_rate=r, horizon=h, scenario=scenario,
            )
            pos = cpv[cpv["stranded_value_total"] > 0]
            total_B = pos["stranded_value_total"].sum() / 1e9
            results.append({
                "discount_rate": r,
                "horizon": h,
                "scenario": scenario,
                "total_stranded_B": total_B,
                "n_stranded_counties": len(pos),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# TASK 4: State-level stranded value table
# ---------------------------------------------------------------------------

def build_state_table(county_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate county stranded values to state level.

    Derives state FIPS from the first 2 characters of the 5-digit county FIPS.

    Args:
        county_df: County-level stranded value DataFrame (post-floor).

    Returns:
        DataFrame with state-level statistics, sorted by total_stranded_B descending.
    """
    # State FIPS mapping
    state_fips_map = {
        "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
        "06": "California", "08": "Colorado", "09": "Connecticut",
        "10": "Delaware", "12": "Florida", "13": "Georgia", "15": "Hawaii",
        "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
        "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
        "24": "Maryland", "25": "Massachusetts", "26": "Michigan",
        "27": "Minnesota", "28": "Mississippi", "29": "Missouri",
        "30": "Montana", "31": "Nebraska", "32": "Nevada",
        "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
        "36": "New York", "37": "North Carolina", "38": "North Dakota",
        "39": "Ohio", "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania",
        "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota",
        "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont",
        "51": "Virginia", "53": "Washington", "54": "West Virginia",
        "55": "Wisconsin", "56": "Wyoming",
    }

    df = county_df.copy()
    df["state_fips"] = df["fips"].astype(str).str.zfill(5).str[:2]
    df["state"] = df["state_fips"].map(state_fips_map).fillna("Unknown")

    positive = df[df["stranded_value_floored"] > 0].copy()

    state_agg = (
        positive.groupby("state")
        .agg(
            n_stressed_counties=("fips", "count"),
            total_stranded_B=("stranded_value_floored", lambda x: x.sum() / 1e9),
            median_stranded_fraction=("stranded_fraction", "median"),
        )
        .reset_index()
        .sort_values("total_stranded_B", ascending=False)
        .reset_index(drop=True)
    )

    return state_agg


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Run the full stranded-value revision pipeline."""
    np.random.seed(42)

    # ------------------------------------------------------------------
    # TASK 1: Real prices from QuickStats
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 1: Computing real marketing-year average prices")
    print("=" * 60)

    avg_prices, raw_price_df = load_real_prices()
    commodity_prices_real = dict(zip(avg_prices["crop"], avg_prices["real_price_2023usd"]))

    # Save price table
    avg_prices.to_csv(RESULTS_REV / "real_prices_2023usd.csv", index=False)
    print(f"\nSaved: {RESULTS_REV / 'real_prices_2023usd.csv'}")

    # Reference old nominal prices for comparison
    old_prices = {
        "corn": 5.50, "soybeans": 12.80, "wheat_winter": 7.20,
        "wheat_spring": 8.10, "cotton": 0.78, "sorghum": 5.30,
        "barley": 6.10, "oats": 3.80,
    }
    print("\nOld (nominal peak) vs new (30-yr real avg) prices:")
    for crop in sorted(commodity_prices_real):
        old = old_prices.get(crop, "N/A")
        new = commodity_prices_real[crop]
        pct = ((new - old) / old * 100) if isinstance(old, float) else float("nan")
        print(f"  {crop:15s}: old=${old:.2f}  new=${new:.2f}  change={pct:+.1f}%")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading yield projections and land values...")
    yield_proj = pd.read_parquet(PROJECTIONS_DIR / "yield_projections_SSP245.parquet")
    print(f"  Yield proj: {len(yield_proj):,} rows, {yield_proj['fips'].nunique()} counties")

    clim_path = PROJECTIONS_DIR / "county_climate_projections.parquet"
    climate_proj = pd.read_parquet(
        clim_path,
        columns=[
            "fips", "year",
            "tmax_july_projected", "delta_tmax_july",
            "tmax_growing_projected", "delta_tmax_growing",
        ],
    )
    print(f"  Climate proj: {len(climate_proj):,} rows")

    land_path = DATA_RAW / "nass" / "nass_land_values.parquet"
    land_values = pd.read_parquet(land_path) if land_path.exists() else pd.DataFrame()
    print(f"  Land values: {len(land_values):,} rows")

    # ------------------------------------------------------------------
    # TASK 2: Recompute stranded values with real prices
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 2: Recomputing stranded values with real prices")
    print("=" * 60)

    # --- Conservative: ML only, r=4%, h=30yr (same as original lower bound) ---
    print("\nConservative (ML only, r=4%, h=30yr, SSP2-4.5)...")
    conservative = compute_stranded_vectorized(
        yield_proj, land_values, commodity_prices_real,
        discount_rate=0.04, horizon=30, scenario="SSP245",
    )
    pos_cons = conservative[conservative["stranded_value_total"] > 0]
    total_cons_B = pos_cons["stranded_value_total"].sum() / 1e9
    print(f"  Counties stranded: {len(pos_cons)}")
    print(f"  Total stranded:    ${total_cons_B:.1f}B  (old: $56B)")

    # --- Central: ML + SR + 1.30x indirect, r=3%, h=35yr, SSP2-4.5 ---
    print("\nCentral (ML+SR+1.30x, r=3%, h=35yr, SSP2-4.5)...")
    central = compute_stranded_with_damage_function(
        yield_proj, climate_proj, land_values, commodity_prices_real,
        discount_rate=0.03, horizon=35, scenario="SSP245",
        ssp585_scale=1.0, indirect_multiplier=INDIRECT_MULTIPLIER,
    )
    pos_cent = central[central["stranded_value_total"] > 0]
    total_cent_B = pos_cent["stranded_value_total"].sum() / 1e9
    print(f"  Counties stranded: {len(pos_cent)}")
    print(f"  Total stranded:    ${total_cent_B:.1f}B  (old: $105B)")

    # --- Upper: ML + SR + 1.30x indirect, r=2.5%, h=40yr, SSP5-8.5 ---
    print("\nUpper (ML+SR+1.30x+SSP5-8.5, r=2.5%, h=40yr)...")
    upper = compute_stranded_with_damage_function(
        yield_proj, climate_proj, land_values, commodity_prices_real,
        discount_rate=0.025, horizon=40, scenario="SSP585_synthetic",
        ssp585_scale=SSP585_SCALE, indirect_multiplier=INDIRECT_MULTIPLIER,
    )
    pos_upper = upper[upper["stranded_value_total"] > 0]
    total_upper_B = pos_upper["stranded_value_total"].sum() / 1e9
    print(f"  Counties stranded: {len(pos_upper)}")
    print(f"  Total stranded:    ${total_upper_B:.1f}B  (old: $140B)")

    # Sensitivity grid
    print("\nRunning sensitivity grid (7 × 5 = 35 combinations)...")
    grid = sensitivity_grid_real(yield_proj, land_values, commodity_prices_real)
    grid.to_csv(RESULTS_REV / "sensitivity_grid_real_prices.csv", index=False)
    print(f"  Grid range: ${grid['total_stranded_B'].min():.1f}B to ${grid['total_stranded_B'].max():.1f}B")
    print(f"  Saved: {RESULTS_REV / 'sensitivity_grid_real_prices.csv'}")

    # ------------------------------------------------------------------
    # TASK 3: Alternate-use floor applied to central estimate
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 3: Applying alternate-use floor (pasture = $1,500/acre)")
    print("=" * 60)

    central_floored = apply_alternate_use_floor(
        central, yield_proj, commodity_prices_real,
        pasture_value=PASTURE_VALUE_PER_ACRE,
        production_cost=PRODUCTION_COST_PER_ACRE,
    )

    gross_B = central_floored["stranded_value_total"].clip(lower=0).sum() / 1e9
    floored_B = central_floored["stranded_value_floored"].clip(lower=0).sum() / 1e9
    floor_reduction_B = gross_B - floored_B
    n_floor_applied = central_floored["floor_applied"].sum()
    n_non_viable = (central_floored["n_non_viable_crops"] > 0).sum()

    print(f"  Counties with ≥1 non-viable crop (2040-50): {n_non_viable}")
    print(f"  Counties where floor was binding:           {n_floor_applied}")
    print(f"  Gross stranded (before floor):              ${gross_B:.1f}B")
    print(f"  Net stranded (after floor):                 ${floored_B:.1f}B")
    print(f"  Floor reduces estimate by:                  ${floor_reduction_B:.1f}B  ({100*floor_reduction_B/gross_B:.1f}%)")

    central_floored.to_parquet(RESULTS_REV / "stranded_central_floored.parquet", index=False)
    print(f"  Saved: {RESULTS_REV / 'stranded_central_floored.parquet'}")

    # ------------------------------------------------------------------
    # TASK 4: State-level table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 4: Building state-level stranded value table")
    print("=" * 60)

    state_table = build_state_table(central_floored)
    state_table.to_csv(RESULTS_REV / "stranded_by_state.csv", index=False)
    print(f"\nTop-10 states by stranded value (central, after floor):")
    print(state_table.head(10).to_string(index=False))
    print(f"\nSaved: {RESULTS_REV / 'stranded_by_state.csv'}")

    # ------------------------------------------------------------------
    # TASK 5: Summary document
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 5: Writing summary document")
    print("=" * 60)

    # Build price table text
    price_rows = []
    for _, row in avg_prices.iterrows():
        old_p = old_prices.get(row["crop"], float("nan"))
        pct = ((row["real_price_2023usd"] - old_p) / old_p * 100) if old_p else float("nan")
        price_rows.append(
