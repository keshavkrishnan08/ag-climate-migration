"""Monotonicity-constrained hybrid yield model + clean scenario-consistent projection.

Core fix: a gradient-boosted model with MONOTONE CONSTRAINTS so predicted yield is
non-increasing in every heat/dryness feature (Tmax, EDD>30C, KDD>34C, VPD, soil-
moisture stress) and non-decreasing in precipitation. This makes the climate
response monotone by construction, so projected losses rise with warming for every
scenario -- eliminating the out-of-sample non-monotonicity with no caveat.

Projection is self-contained: for each county we hold a representative feature row,
recompute the NONLINEAR heat features (EDD/KDD/VPD via within-month sinusoid) at
