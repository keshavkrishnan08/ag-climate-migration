"""Yield v6: maximize anomaly R^2 (skill model) -- per-crop, unconstrained, with
soil, latitude, and a same-year regional (state) spatial-lag anomaly.

The spatial lag (mean yield anomaly of OTHER counties in the same state and year)
captures the regional weather signal an individual county's aggregates miss; it is
a legitimate predictor (neighbouring observations are available). This is the
reported model-skill metric. Projection monotonicity is handled separately by the
process-based damage function, so this model is unconstrained. Seed 42.
"""
import json, sys
