"""
build_published_dataset.py

Assemble 6 publication-ready CSVs from the ag_migration pipeline outputs.
All monetary values in 2023 USD. Temperatures in °F (climate files).
Yields in bushels per acre.

Run from repo root:
    python src/build_published_dataset.py

Outputs (data/published_dataset/):
    county_yield_projections.csv
    county_climate_projections.csv
    county_stranded_assets.csv
    county_decline_indicators.csv
    county_insurance_mispricing.csv
    county_opportunity_frontier.csv
"""

import os
