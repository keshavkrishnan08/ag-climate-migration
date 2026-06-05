"""Audit improvement C: a defensible cotton-only yield model + irrigation diagnosis.

Cotton anomaly R^2 sits near zero in the pooled model. The hypothesis a hostile
reviewer would demand we test: cotton's interannual yield is governed by
irrigation and soil, not by rainfed climate variability, so climate features
cannot explain it where water is supplied artificially. We test this directly.

Steps:
1. Build a cotton-ONLY model on the full engineered climate feature set
   (modern agro-climatic + drought-trajectory) plus an NCCPI soil-quality proxy
   (county yield ceiling / national ceiling) and a latitude proxy. Held-out
   2013-2023 R^2 / Spearman. This is the best honest climate-only cotton number.
2. Classify counties as IRRIGATION-DOMINATED vs RAINFED using an observable,
   non-circular signal: the historical sensitivity of county yield to growing-
   season PDSI (computed on the TRAIN period only). Irrigated counties show
   near-zero yield-PDSI coupling; rainfed counties show strong coupling.
3. Re-evaluate the model's held-out skill SEPARATELY on rainfed vs irrigated
   cotton counties. If climate skill is concentrated in rainfed counties, that
   is direct evidence the cotton ceiling is irrigation-driven, not a model defect.

