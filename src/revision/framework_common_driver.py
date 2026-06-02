"""Test the UNIFYING claim correctly: not as a causal chain among the four findings
(that hypothesis failed: stranded->decline p=0.50, insurance->switching wrong sign),
but as a COMMON CAUSE. One institutional failure = backward-looking valuation of a
forward-moving climate. The testable implication: a single PHYSICAL forward-climate-
exposure signal (which backward-looking institutions do not price) should drive all
four channel outcomes, with signs coherent across channels.

Driver: forward climate exposure per county (change in extreme degree-days; projected
Schlenker-Roberts yield penalty). Physical, upstream of every dollar channel, so the
test is not mechanical for the three non-definitional channels.

Channels:
  C1 Stranded value per acre        (capitalization gap)           expect +
  C2 Insurance net underpricing     (APH lags rising risk)         expect +
  C3 Rural-decline indicator count  (farming-dependent counties)   expect +
  C4 Northern frontier opportunity  (warming gain in the north)    expect + with GDD gain

If one exposure variable predicts C1-C4 with coherent signs, the four are parallel
consequences of one mispriced climate signal -- the honest form of 'one mechanism'.
Seed 42; writes only to results/revision/.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
np.random.seed(42)
OUT = Path("results/revision")

def hc1(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    cov = XtXi @ ((X * (r ** 2)[:, None]).T @ X) @ XtXi * (len(y) / (len(y) - X.shape[1]))
    se = np.sqrt(np.diag(cov))
    return b, se

def z(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std()

def reg(df, yname, xname, label, expect):
    d = df[[yname, xname]].replace([np.inf, -np.inf], np.nan).dropna()
    X = np.column_stack([np.ones(len(d)), z(d[xname]).values])
    b, se = hc1(d[yname].astype(float).values, X)
    t = b[1] / se[1]
    p = 2 * (1 - stats.norm.cdf(abs(t)))
    sign_ok = (b[1] > 0) == (expect > 0)
    return {"channel": label, "beta_per_1sd": float(b[1]), "se": float(se[1]),
            "p": float(p), "n": int(len(d)), "sign_as_predicted": bool(sign_ok)}

