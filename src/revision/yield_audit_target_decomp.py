"""Audit improvement E (the hostile-reviewer test): is v7's +0.13 R^2 gain real
predictive skill, or an artifact of switching the target metric?

The paper attributes most of the jump from R^2=0.23 to R^2=0.41 to predicting the
percentage deviation from trend instead of the z-scored anomaly. A hostile
reviewer's first objection: "R^2 on a different target is not comparable -- you
may simply have chosen a target with a more favorable variance structure, not a
better model." We settle this with a clean 2x2 factorial that holds the
EVALUATION target fixed within each cell:

  Factor 1 (architecture/features): AGG = growing-season aggregates (the old
            feature set) vs SPEC = temperature-exposure spectrum + per-crop.
  Factor 2 (training target):       Z = z-scored anomaly vs PCT = % deviation.

For every cell we ALSO map predictions onto BOTH common scales and report R^2 on
each, so the four models are compared on identical yardsticks:
  - R^2 on the z-scored anomaly (the conservative, old yardstick)
  - R^2 on the % deviation (the new yardstick the paper reports)
Mapping: a %-deviation prediction is converted to a yield level (trend*(1+dev))
and then z-scored with the same county-crop mean/sd used to build the anomaly,
