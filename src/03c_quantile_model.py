"""Phase 3C: Quantile Regression Forest for Extreme Event Tail Risk.

The mean yield model underpredicts extreme events — the 2012 drought anomaly
prediction of -0.11 is far too mild vs. the observed mean of -0.94 across
corn counties. Tail risk matters for stranded asset estimates, which depend
on correctly capturing downside scenarios.

This script:
  1. Trains a LightGBM quantile model at alpha=0.10 (Q10) using same features
     and temporal split as the existing mean model.
