"""Revision: restrict the rural-decline analysis to farming-dependent counties
and replace coincident indicator COUNTS with MARGINAL EFFECTS (Reviewer 1 #2, Fig 7B).

Reviewer 1 raised three linked points:
  (a) The yield -> outmigration mechanism may be externally invalid for the
      diversified US economy; only counties with a high farm share of income
      should be expected to show it. Restrict to that subset.
  (b) Figure 7B (coincident counts of decline indicators) is misleading because
      the indicators are merely concurrent; report the MARGINAL effect of yield
      decline on each indicator instead.
  (c) The weather IV likely violates the exclusion restriction (weather affects
      migration via amenity / winter-mildness channels, Rappaport 2007), so the
      defensible object is the direct chain yield -> farm income -> local fiscal
      capacity, not a causal migration elasticity.

This script:
  1. Flags the 444 ERS farming-dependent counties (Type_2015_Farming_NO=1).
  2. Recomputes the share of counties with >=4 decline indicators within the
     farming-dependent subset vs the rest (the restricted observational result).
  3. Estimates the marginal effect of a 1-SD adverse yield anomaly on each
