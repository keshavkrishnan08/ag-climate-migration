"""Phase 5B: Community collapse cascade.

Uses econometric literature's estimated elasticities to propagate yield decline
into community-level outcomes. Not a standalone ML model — it uses ML yield
projections as input to a structured cascade.

Cascade structure (PRD Section 5.3 & 7B):
    Step 1: Yield decline → Farm income decline
            ΔIncome = Σ_crops [ΔYield × Acres × Price × (1 - InsuranceOffset)]
    Step 2: Farm income → Rural outmigration
