"""Audit improvement C: a defensible cotton-only yield model + irrigation diagnosis.

Cotton anomaly R^2 sits near zero in the pooled model. The hypothesis a hostile
reviewer would demand we test: cotton's interannual yield is governed by
irrigation and soil, not by rainfed climate variability, so climate features
cannot explain it where water is supplied artificially. We test this directly.

Steps:
1. Build a cotton-ONLY model on the full engineered climate feature set
   (modern agro-climatic + drought-trajectory) plus an NCCPI soil-quality proxy
