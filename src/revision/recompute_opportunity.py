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
