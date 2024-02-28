"""Phase 5C: Hedonic farmland valuation with soil productivity controls.

Reviewer concern: the baseline hedonic (06b) uses climate + socioeconomic
controls but no soil quality control. Omitting soil quality may bias the
climate coefficient if soil quality correlates with both temperature and land
values (e.g., rich soils in the Corn Belt → high values AND moderate temps).

This script:
1. Constructs a county-level NCCPI proxy (soil productivity index) from
   maximum observed corn yield 1950–2023, normalized to [0,1]. Max yield
   captures soil potential under optimal weather rather than average climate
   response (Schlenker & Roberts 2009 precedent).
2. Adds an ERS HiAmenity binary control from the USDA ERS Rural Atlas 2024,
   directly addressing the amenity channel.
3. Re-estimates the hedonic regression with both new controls.
4. Compares: climate coefficients, R², and stranded asset estimate
   with vs without soil controls.

Model (baseline, 06b):
    log(V) = β₀ + β_T·T + β_T²·T² + β_P·P + β_pop·log(pop)
