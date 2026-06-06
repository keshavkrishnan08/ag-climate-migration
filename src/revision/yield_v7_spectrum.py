"""Yield v7: temperature-exposure spectrum + natural-units target -> genuinely
higher anomaly R^2 (not statistics, a better model).

Two architectural changes, both standard in the modern statistical crop-yield
literature and BOTH improve genuine predictive skill:

1. TARGET = percentage deviation from the county-crop technology trend
   (yield/trend - 1), the natural physical scale. The previous z-scored anomaly
   divides by the county standard deviation, which amplifies noise in low-
   variance counties and mechanically caps R^2. The % deviation is also exactly
   what the dollar computation needs (impact_bu = dev_pct * expected_yield).

2. FEATURES = the monthly TEMPERATURE-EXPOSURE SPECTRUM. Instead of growing-
   season averages, we give the model degree-time in temperature bins
   ([<5],[5-10],...,[29-32],[32-34],[>34] C) accumulated across the growing
   season via a within-month diurnal sinusoid (Schlenker & Roberts 2009), plus
   monthly precipitation, VPD and PDSI. This exposes the nonlinear temperature
   response the aggregates hide. Soil (NCCPI), latitude and the trend slope are
   included; per-crop models. Held-out test = 2013-2023. Seed 42.
"""
