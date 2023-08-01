"""Phase 3C: Quantile Regression Forest for Extreme Event Tail Risk.

The mean yield model underpredicts extreme events — the 2012 drought anomaly
prediction of -0.11 is far too mild vs. the observed mean of -0.94 across
corn counties. Tail risk matters for stranded asset estimates, which depend
on correctly capturing downside scenarios.

This script:
  1. Trains a LightGBM quantile model at alpha=0.10 (Q10) using same features
     and temporal split as the existing mean model.
  2. Diagnoses the 2012 drought: Q10 prediction should be much more negative
     than the mean model's -0.11.
  3. Computes a tail risk premium per county (mean - Q10 gap) and adds it as
     an additional stranded asset component.

Temporal split (per CLAUDE.md):
    train  : year <= 2009
    val    : 2010-2016  (used as final training cutoff for test evaluation)
    test   : 2017-2023

