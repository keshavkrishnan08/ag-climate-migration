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
