"""Revision: insurance mispricing with a REAL-TIME ROLLING APH simulation.

Reviewer 2 (Major Concerns 2 & 3) showed the original mispricing figure
($5.9 B yr-1) overstates the policy-relevant number because it compared a
forward 2040-2050 yield projection against a FROZEN historical APH baseline
(src/08_insurance.py uses yield_baseline as a constant). In reality, Actual
Production History (APH) is a 4-to-10-year rolling mean that updates every year,
so it mechanically absorbs most of a smooth climate trend at a ~5-year lag.

This script rebuilds the estimate to answer the reviewer precisely:

  (R2 #3a) Simulate the APH update mechanism: compare the forward projected
           yield against an APH-equivalent rolling mean computed in real time
           as years progress, instead of a frozen baseline.

  (R2 #3b) Report mispricing net of Trend-Adjusted Yield (TAY) endorsements at
           current participation, and bound the Yield-Exclusion (YE) effect.

  (R2 #3c) Decompose the gross gap into:
             (i)   what the rolling APH window absorbs mechanically,
