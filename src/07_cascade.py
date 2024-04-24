"""Phase 5B: Community collapse cascade.

Uses econometric literature's estimated elasticities to propagate yield decline
into community-level outcomes. Not a standalone ML model — it uses ML yield
projections as input to a structured cascade.

Cascade structure (PRD Section 5.3 & 7B):
    Step 1: Yield decline → Farm income decline
            ΔIncome = Σ_crops [ΔYield × Acres × Price × (1 - InsuranceOffset)]
    Step 2: Farm income → Rural outmigration
            ΔPop = elasticity × ΔIncome% (lagged 3 years)
    Step 3: Outmigration → School enrollment decline
            ΔEnrollment = -0.25 × ΔPop (empirical from NCES, contemporaneous)
    Step 4: Population → Hospital viability
            Closure threshold: county pop < 15,000
    Step 5: Farm income + Population → Tax base
            ΔTaxBase = ΔFarmIncome × 0.35 + ΔPop × AvgPerCapitaTax
    Step 6: Tax base → Infrastructure
            ΔRoadCondition = f(ΔTaxBase) (lagged 5 years)
    Step 7: Infrastructure → Further yield loss (FEEDBACK LOOP)
            Feedback_multiplier = 0.08 per σ decline in infrastructure

Tipping point: county crosses when ALL FOUR conditions met simultaneously:
    1. Population below hospital threshold
    2. School enrollment below closure threshold
    3. Infrastructure feedback accelerating yield loss
    4. Outmigration > 2× in-migration

Target finding: 300 counties cross tipping point before 2040 under RCP 4.5.

Reviewer Fix 4: Re-estimate migration elasticity via IV on 2000-2020 data.
Dual calibration (Reviewer Fix — Issue 2):
    Calibration A: Own IV estimate β=-0.003 (p=0.019, F=1184) — PRIMARY
    Calibration B: Feng et al. (2010) β=-0.17 — SENSITIVITY
    Both are reported so reviewers can evaluate the 57x difference in magnitude.
"""

import os
import sys
from pathlib import Path
