"""Formally model endogenous coverage selection (Reviewer 2, Major 2).

R2: "farmers in climate-stressed counties may rationally select higher coverage
because indemnity probability is higher, biasing the implicit-transfer estimate in
a direction the uniform calculation cannot capture."

We do two things:
(1) TEST the relationship: regress the acreage-weighted coverage election on a
    county climate-stress measure (projected yield decline), with crop fixed
    effects and acreage weights. A positive, significant coefficient confirms
