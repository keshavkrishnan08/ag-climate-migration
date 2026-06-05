"""Revision: yield levels-R^2 and a properly propagated DCF confidence interval
(Reviewer 2 #1).

Two points:

(A) The reviewer's R^2 > 0.5 benchmark refers to models that predict yield
    LEVELS (which include the dominant technology trend and county effects).
    Our headline R^2 = 0.22 is on the much harder z-scored, detrended ANOMALY.
    On a true out-of-sample levels hindcast (2013-2023), the same model explains
    the great majority of yield variance. We compute that here.

(B) The Monte Carlo CI of [$58, $63 B] was implausibly tight because it
    propagated only IDIOSYNCRATIC county errors, which cancel in aggregation.
    A defensible interval must also carry (i) spatially correlated prediction
    error (regional weather events), and (ii) GCM ensemble spread. We propagate
    all three and report the decomposition so the widening is transparent.

Seed 42. Writes only to results/revision/.
"""

