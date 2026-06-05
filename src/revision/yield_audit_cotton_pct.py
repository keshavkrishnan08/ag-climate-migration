"""Audit improvement C2: best-honest cotton model on the %-deviation target
(matching the paper's v7 yardstick), with the irrigation diagnosis.

The paper reports cotton deviation R^2 = 0.164 (Spearman 0.483). We test whether
a cotton-DEDICATED model on the same %-deviation target, enriched with the
drought-trajectory features and a soil/irrigation proxy, beats that, and we
quantify how much of cotton's irreducible variance is irrigation.

Approach:
1. Build the v7 panel (spectrum + monthly precip/PDSI/VPD, NCCPI, latitude,
