"""Audit improvement D: best combined model + a learning-curve ceiling test.

Two deliverables a hostile methods reviewer would want:

1. BEST HONEST POOLED MODEL. The drought-trajectory features (audit A) gave the
   single biggest gain (R^2 0.227 -> 0.234). Here we lock in that feature set with
   light hyper-parameter tuning and report the final pooled + per-crop held-out
   anomaly R^2 / Spearman, plus confirm the WITHIN-CROP LEVELS R^2 (the number the
   paper also reports) is not degraded.

