"""Phase 5A: Stranded agricultural asset valuation.

Computes the present discounted value gap between farmland valued under
a no-climate-change trajectory and farmland valued under projected climate.

    Stranded value = PV(income under tech trend only) - PV(income under tech + climate)
    Positive = county loses value due to climate (stranded asset)
    Negative = county gains value (climate benefit)

Reviewer Fix 3: Sensitivity grid (discount 2-8% x horizon 20-40yr) + cap rate method.
