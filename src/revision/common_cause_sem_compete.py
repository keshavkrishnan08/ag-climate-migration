"""E51: Compete the single-factor SEM against two alternative structures.

Reviewer concern: with 4 channels and 2 degrees of freedom, single-factor fit
is nearly guaranteed; it cannot distinguish a genuine common institutional
cause from shared exposure to a thermal driver (e.g., July Tmax). We compete:

  M0 (saturated)        : correlated 4-channel covariance, df = 0
  M1 (single factor)    : 4 loadings, df = 2 (the baseline reported)
  M2 (two-factor)       : 2 latents = {institutional pricing} and {direct
                          thermal exposure}, with C1, C2 loading on F1 and
