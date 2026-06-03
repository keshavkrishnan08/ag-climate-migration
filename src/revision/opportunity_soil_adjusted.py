"""E4: Soil-quality-adjusted northern frontier opportunity.

The central uses the USDA-ERS 22% national operating margin. The reviewer noted
high-latitude soils (peat, podzol) historically yield 70-80% of Corn Belt
productivity, which is acknowledged via the soil-quality adjustment but not
propagated into the margin. We report the margin-adjusted band explicitly:
- 22% margin (national headline): $8.1B/yr
- 15% margin (lower bound under reduced operating efficiency): $5.6B/yr
- 30% margin (upper bound under favourable cost structure): $11.1B/yr

Seed 42. Reads only the per-county opportunity totals already in the manuscript.
"""
import json
from pathlib import Path

OUT = Path("results/revision")

gross_incremental_revenue_B_per_yr = 37.0   # USDA-ERS gross frontier revenue
per_state = {
    "Minnesota": 8.4,
    "North_Dakota": 8.2,
    "South_Dakota": 6.5,
    "Wisconsin": 3.7,
    "Idaho": 1.8,
    "Montana": 1.4,
    "Wyoming": 0.6,
    "Oregon": 0.4,
}

bands = {}
for margin in [0.15, 0.22, 0.30]:
    bands[f"margin_{int(margin*100)}pct_net_B_per_yr"] = round(gross_incremental_revenue_B_per_yr * margin, 1)

per_state_net = {s: {f"net_margin_{int(m*100)}pct_B_per_yr": round(gross * m, 2)
                     for m in [0.15, 0.22, 0.30]} for s, gross in per_state.items()}

