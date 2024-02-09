"""Phase 5B: Mendelsohn-Nordhaus-Schlenker hedonic farmland valuation.

Cross-sectional hedonic regression of observed farmland values on climate
variables (Mendelsohn, Nordhaus & Shaw 1994; Schlenker, Hanemann & Fisher
2005, 2006). No discount rate required. Captures ALL value channels —
crop yields, amenity value, water, livestock, specialty crops — via
market-revealed land prices.

Model:
    log(land_value) = β₀ + β₁·tmax_july + β₂·tmax_july² + β₃·precip_growing
