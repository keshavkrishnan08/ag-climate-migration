"""Formally model endogenous coverage selection (Reviewer 2, Major 2).

R2: "farmers in climate-stressed counties may rationally select higher coverage
because indemnity probability is higher, biasing the implicit-transfer estimate in
a direction the uniform calculation cannot capture."

We do two things:
(1) TEST the relationship: regress the acreage-weighted coverage election on a
    county climate-stress measure (projected yield decline), with crop fixed
    effects and acreage weights. A positive, significant coefficient confirms
    stressed counties up-select coverage.
(2) QUANTIFY the bias: recompute the rolling-APH residual mispricing and implicit
    transfer using each county-crop's ACTUAL elected coverage (endogenous) versus a
    counterfactual UNIFORM coverage (national mean). The difference is the part of
    the transfer that a uniform-coverage calculation cannot capture.

Seed 42; writes only to results/revision/.
"""
import json, sys
from pathlib import Path
