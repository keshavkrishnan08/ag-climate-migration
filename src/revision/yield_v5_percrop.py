"""Yield v5: per-crop monotone models with soil, latitude, and autoregressive
features -- raises anomaly R^2 without dropping any crop.

Additions over v4:
  * PER-CROP models (climate-yield response differs by crop; pooled dilutes it).
  * Soil quality proxy (NCCPI-style: county max historical yield / national max).
  * County latitude (centroid, USDA gazetteer).
  * Autoregressive prior-year own anomaly (AR1) -- persistence from soil moisture
    carryover and management; known at planting, set to 0 under projection.
  * Monotone climate constraints retained (heat/dry -1, precip +1) so the model
