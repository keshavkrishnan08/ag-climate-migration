"""Audit improvement C2: best-honest cotton model on the %-deviation target
(matching the paper's v7 yardstick), with the irrigation diagnosis.

The paper reports cotton deviation R^2 = 0.164 (Spearman 0.483). We test whether
a cotton-DEDICATED model on the same %-deviation target, enriched with the
drought-trajectory features and a soil/irrigation proxy, beats that, and we
quantify how much of cotton's irreducible variance is irrigation.

Approach:
1. Build the v7 panel (spectrum + monthly precip/PDSI/VPD, NCCPI, latitude,
   %-deviation target) and ADD the drought-trajectory features (dry-run,
   PDSI slope, deficit integral, timing, early-late contrast).
2. Train a cotton-only LightGBM with mild regularization. Report deviation R^2 /
   Spearman vs the paper's 0.164.
3. Split cotton counties into rainfed vs irrigated by TRAIN-period yield-PDSI
   coupling and report skill separately -- the evidence that the cotton ceiling
   is an irrigation effect, not a model defect.

Split: train<=2012, test 2013-2023. Seed 42.
"""
