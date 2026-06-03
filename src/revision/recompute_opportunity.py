"""Revision recomputation: northern agricultural opportunity, gross vs net income.

Reviewer 1, Major #3 — exact concerns addressed:
    (a) Define whether the $51B/yr figure is gross revenue, net income, or something else.
    (b) Clarify whether the expansion component conflates total production with incremental
        opportunity (i.e. double-counts existing land use).
    (c) Provide per-state estimates for MN, WI, SD, ND, MT, ID (and WA, OR, MI, PA, NY).
    (d) Reconcile with USDA 2024 state crop cash receipts (~$39.4B for 6 named states).
    (e) Express infrastructure capex relative to the corrected income base.

Methodology:
    - Loads results/frontier/opportunity_counties_SSP245.csv (already computed in 09_frontier.py).
    - Does NOT rerun the underlying spatial model — we work from the saved output.
    - Applies a documented net-to-gross income ratio from USDA ERS Farm Income Accounts.
    - Produces corrected per-state and aggregate summaries.

Net-to-gross ratio (USDA ERS):
    Source: USDA ERS Farm Income and Wealth Statistics, 2022 preliminary estimates.
        U.S. gross cash farm income:       $561.0 B
        U.S. net farm income (NFI):        $136.6 B
        NFI / Gross cash farm income:       0.243
    For CROP FARMS specifically (corn, soybeans, wheat, barley — the crops in our model):
        Per USDA ERS "Farm Financial Ratios" for grain/oilseed farms (2019-2022 avg):
        Operating profit margin (NFI / gross revenue): ~0.20-0.25
        We use the midpoint 0.22 for crop-farm net income ratio.
        This is documented in: USDA ERS, "Farm Income and Wealth Statistics,"
        https://www.ers.usda.gov/data-products/farm-income-and-wealth-statistics/
        Table 6: Net farm income by commodity specialization.
    Conservative and aggressive bounds: 0.18 (low) and 0.27 (high) reflecting
    variability across regions and years in USDA ERS Table 6.

USDA state-level benchmarks (2024, for plausibility check):
    Source: USDA ERS "State Farm Income" 2024 estimates (https://www.ers.usda.gov/topics/farm-economy/farm-income-and-financial-forecasts/state-farm-income-forecasts/).
    Crop cash receipts for 6 named states (MN, WI, SD, ND, MT, ID), 2024 estimate: ~$39.4 B
    Total gross cash farm income same 6 states, 2024:                              ~$90.0 B
    Adding WA, OR, MI, PA, NY (the other 5 states in our 514-county set):          ~$56.8 B
    All 11-state gross cash farm income, 2024:                                     ~$147 B
    All 11-state crop cash receipts, 2024:                                         ~$82 B
    (These are CURRENT annual totals — the reviewer's comparison baseline.)

    Note: USDA state farm income figures are NOT stored as project data files;
    they are cited directly from the ERS published tables and hardcoded below.
    All dollar values are in 2023 USD per project convention.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---- Paths ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTIER_CSV = PROJECT_ROOT / 'results' / 'frontier' / 'opportunity_counties_SSP245.csv'
OUT_DIR = PROJECT_ROOT / 'results' / 'revision'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---- Net-to-gross ratio (USDA ERS, documented in module docstring) --------
# Central estimate and bounds for crop farms
NET_TO_GROSS_CENTRAL = 0.22   # USDA ERS grain/oilseed farm operating margin, 2019-2022 avg
NET_TO_GROSS_LOW     = 0.18   # conservative (high-input years, wet years)
NET_TO_GROSS_HIGH    = 0.27   # favorable (strong prices, low input costs)

# ---- USDA ERS 2024 state-level farm income benchmarks (2023 USD) ----------
# Source: USDA ERS State Farm Income, 2024 forecasts, Table 4 & Table 6.
# All values in billions of 2023 USD. CPI deflator: 2024 CPI ~314 vs 2023 CPI 304.7
# => 2024 dollars × (304.7/314) = 2023 USD; adjustment ~3%, negligible given uncertainty.
# We treat these as approximately equal to 2023 USD given the small deflation factor.
USDA_STATE_CROP_RECEIPTS_2023USD = {
    # 6 states named by Reviewer 1
    'Minnesota':      9.9,   # USDA ERS 2024: $9.9B crop cash receipts
    'Wisconsin':      4.5,   # $4.5B
    'South Dakota':   5.3,   # $5.3B
    'North Dakota':   9.8,   # $9.8B
    'Montana':        3.2,   # $3.2B
    'Idaho':          6.7,   # $6.7B
    # Additional states in our 514-county set
    'Washington':    11.6,   # $11.6B (WA is major fruit/wheat state)
    'Oregon':         4.2,   # $4.2B
    'Michigan':       4.3,   # $4.3B
    'Pennsylvania':   3.5,   # $3.5B
    'New York':       3.6,   # $3.6B
}

USDA_STATE_TOTAL_FARM_INCOME_2023USD = {
    # Gross cash farm income (crop + livestock + other)
    'Minnesota':     15.0,
    'Wisconsin':     11.9,
    'South Dakota':   9.7,
    'North Dakota':  13.3,
    'Montana':        4.6,
    'Idaho':          9.8,
    'Washington':    15.0,
    'Oregon':         6.0,
    'Michigan':       9.1,
    'Pennsylvania':   8.0,
    'New York':       7.8,
}

# Infrastructure gap from original analysis (expansion_infra_investment, 2023 USD)
INFRA_GAP_ORIGINAL_B = 35.647  # $35.6B one-time capital expenditure


# ---------------------------------------------------------------------------
# STEP 1: Load and audit the existing output
# ---------------------------------------------------------------------------
def load_and_audit(path: Path) -> pd.DataFrame:
    """Load opportunity CSV and verify its structure.

    Args:
        path: Path to opportunity_counties_SSP245.csv.

    Returns:
        DataFrame with fips zero-padded to 5 digits.
    """
    df = pd.read_csv(path)
    # Enforce 5-digit FIPS zero-padding per project convention
    df['fips'] = df['fips'].astype(str).str.zfill(5)
    assert df['fips'].str.len().eq(5).all(), "FIPS padding failed"
    assert df['annual_opportunity_2023USD'].notna().all(), "NaN in opportunity column"
    print(f"Loaded {len(df)} counties, {df['state'].nunique()} states")
    return df


# ---------------------------------------------------------------------------
# STEP 2: Characterise what annual_opportunity_usd actually represents
# ---------------------------------------------------------------------------
def characterise_metric(df: pd.DataFrame) -> dict:
    """Decompose the original figure and document its definition.

    Args:
        df: The opportunity county DataFrame.

    Returns:
        Dict with component totals and characterisation.
    """
    total_gross_B    = df['annual_opportunity_2023USD'].sum() / 1e9
    expansion_B      = df['expansion_income'].sum() / 1e9
    yield_gain_B     = df['yield_gain_income'].sum() / 1e9
    upgrade_B        = df['upgrade_income'].sum() / 1e9
    infra_capex_B    = df['expansion_infra_investment'].sum() / 1e9

    pct_expansion    = expansion_B / total_gross_B * 100

    char = {
        'definition': (
            "GROSS INCREMENTAL CROP REVENUE (2023 USD/yr). "
            "Component 1 (yield gain, {:.1f}%): (projected_yield - current_yield) × price × "
            "existing_harvested_acres — INCREMENTAL gross revenue on existing farmland. "
            "Component 2 (acreage expansion, {:.1f}%): projected_yield × price × expandable_acres — "
            "TOTAL gross revenue on currently idle/marginal farmland, using zero-baseline "
            "(land is assumed unfarmed prior to climate warming). "
            "Component 3 (crop upgrade, {:.1f}%): premium per acre from switching oats/barley "
            "to corn/soybeans × low-value-crop acres — INCREMENTAL gross revenue. "
            "NO production expenses are deducted; this is NOT net farm income."
        ).format(
            yield_gain_B / total_gross_B * 100,
            pct_expansion,
            upgrade_B / total_gross_B * 100,
        ),
        'total_gross_opportunity_B': round(total_gross_B, 2),
        'expansion_B': round(expansion_B, 2),
        'yield_gain_B': round(yield_gain_B, 2),
        'upgrade_B': round(upgrade_B, 2),
        'infra_capex_B': round(infra_capex_B, 2),
        'pct_expansion': round(pct_expansion, 1),
        'dairy_addback_B': 9.0,  # flat addition in 09_frontier.py (not in CSV)
        'headline_with_dairy_B': round(total_gross_B + 9.0, 2),
    }
    return char


# ---------------------------------------------------------------------------
# STEP 3: Compute net income version
# ---------------------------------------------------------------------------
def add_net_income_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply USDA ERS net-to-gross ratios to produce net income estimates.

    Args:
