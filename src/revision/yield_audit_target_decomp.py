"""Audit improvement E (the hostile-reviewer test): is v7's +0.13 R^2 gain real
predictive skill, or an artifact of switching the target metric?

The paper attributes most of the jump from R^2=0.23 to R^2=0.41 to predicting the
percentage deviation from trend instead of the z-scored anomaly. A hostile
reviewer's first objection: "R^2 on a different target is not comparable -- you
may simply have chosen a target with a more favorable variance structure, not a
better model." We settle this with a clean 2x2 factorial that holds the
EVALUATION target fixed within each cell:

