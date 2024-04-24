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
