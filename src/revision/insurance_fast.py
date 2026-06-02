"""Vectorized insurance mispricing simulation + round-2 robustness battery.

Reimplements the rolling-APH decomposition (insurance_rolling_aph.py) without the
per-county Python loop, so we can sweep robustness dimensions a tough reviewer
would request next:
  * APH window length (4 / 7 / 10 years) -- shorter windows absorb the trend
    faster, reducing the residual.
  * Yield-Exclusion (YE) at participation -- drop the worst year in the window,
    raising APH in disaster-prone counties (works against TAY).
  * Climate scenario (SSP2-4.5 vs SSP3-7.0).
