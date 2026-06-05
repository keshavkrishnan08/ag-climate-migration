"""Audit improvement A: richer drought-trajectory features + Huber-loss target.

Orthogonal to v7 (which changes the target to %-deviation and uses a temperature
exposure spectrum). Here we KEEP the existing z-scored county-detrended anomaly
target -- so the held-out R^2 is DIRECTLY comparable to the v4 number (0.227) the
paper currently reports -- and attack two distinct, defensible levers:

1. RICHER DROUGHT DYNAMICS from the monthly PDSI/precip panel. Yield loss in
   corn/soy/cotton is driven by drought *timing and persistence*, not just the
   season minimum. We add:
