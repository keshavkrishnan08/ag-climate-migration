"""Migration robustness: weak-IV-robust Anderson-Rubin confidence set,
alternative outcomes, and leave-one-crop-out shift-share stability.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from migration_iv_bartik import build_panel, demean2
OUT = ROOT / "results" / "revision"


def high_tercile(panel):
    base = panel.groupby("fips").agg(base_rev=("base_rev", "first")).reset_index()
    pop0 = panel.sort_values("year").groupby("fips")["total_population"].first().rename("pop0").reset_index()
    fi = base.merge(pop0, on="fips"); fi["fi"] = fi["base_rev"] / fi["pop0"].replace(0, np.nan)
    panel = panel.merge(fi[["fips", "fi"]], on="fips", how="left")
    return panel[panel["fi"] >= panel["fi"].quantile(0.67)].copy()


def ar_test(df, y, d, z, ctrls, beta0):
    """Anderson-Rubin: regress (y - beta0*d) on z (+ctrls) with FE; return p on z."""
    cols = [y, d, z] + ctrls
    dd = df.dropna(subset=cols).copy()
    dd = dd[np.all(np.isfinite(dd[cols].values), axis=1)]
    dd["resid_y"] = dd[y] - beta0 * dd[d]
    dm = demean2(dd, ["resid_y", z] + ctrls)
    Y = dm["resid_y_dm"].values
    X = np.column_stack([dm[z + "_dm"].values] + [dm[c + "_dm"].values for c in ctrls])
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    u = Y - X @ b
    bread = np.linalg.inv(X.T @ X); meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in dd.groupby("fips").indices.items():
        Xg = X[idx]; ug = u[idx]; meat += Xg.T @ np.outer(ug, ug) @ Xg
    cov = bread @ meat @ bread
    se = np.sqrt(cov[0, 0]); t = b[0] / se
    return 2 * (1 - stats.norm.cdf(abs(t)))


def main():
    panel = build_panel()
    high = high_tercile(panel)
    ctrls = ["winter_tmin_anom"]

    # Anderson-Rubin 95% CI for beta on pop_growth_3yr
    grid = np.linspace(-0.05, 0.25, 121)
    accept = [b for b in grid if ar_test(high, "pop_growth_3yr", "fid_3yr", "z_bartik", ctrls, b) > 0.05]
    ar_ci = [float(min(accept)), float(max(accept))] if accept else None
