"""Tier-3: rigorously tighten every cited number.

Strategy:
- Migration wild-cluster bootstrap at B=9999 (vs 1999) -> tighter p-value precision
- Depopulation NPV at 1M draws (vs 200k) -> narrower 90% CI
- Stranded CI propagation at 1M draws
- Hedonic with HC3 robust SEs (more conservative than HC1) + cluster-robust on state
- Migration with HC3 + cluster bootstrap CIs
- Yield acreage-weighted Spearman (the valuation-relevant metric)
- Common-cause test with Bonferroni adjustment across 3 channels
- Insurance residual with cluster-robust over crop x year + 95% interval
- All robustness checks done at higher precision

Every output is paired with a STATISTICAL RIGOR delta showing the improvement vs the
prior result (tighter CI, lower p, etc.).

Seed 42; writes results/revision/tier3_tighten.json.
"""
import json, sys
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
np.random.seed(42)
OUT = Path("results/revision")
sys.path.insert(0, "src/revision")
out = {}

def jl(n):
    p = OUT / n
    return json.load(open(p)) if p.exists() else None

# ============================================================
# T1. Migration wild-cluster bootstrap at B=9999 (vs B=1999)
# ============================================================
print("[T1] Migration wild-cluster bootstrap at B=9999...")
try:
    from migration_iv_bartik import build_panel
    from migration_primeage_panel import within
    prime = pd.read_parquet(OUT / "prime_age_pop.parquet")
    prime["fips"] = prime["fips"].astype(str).str.zfill(5)
    prime = prime.sort_values(["fips", "year"])
    prime["g5"] = prime.groupby("fips")["prime"].transform(lambda s: (s.shift(-5) / s - 1))
    panel = build_panel()
    p = panel.merge(prime[["fips", "year", "g5"]], on=["fips", "year"], how="inner")
    fd = p[p["farm_dependent"] == 1].dropna(subset=["g5", "farm_income_dev", "z_bartik"]).copy()
    fd = within(fd, ["g5", "farm_income_dev", "z_bartik"])
    Y = fd["g5_w"].values; D = fd["farm_income_dev_w"].values; Z = fd["z_bartik_w"].values
    g = fd["fips"].values; clusters = np.unique(g); G = len(clusters)
    rows_by = {c: np.where(g == c)[0] for c in clusters}
    def iv_beta(Y, D, Z): return (Z @ Y) / (Z @ D)
    beta_hat = iv_beta(Y, D, Z)
    u = Y - D * beta_hat
    ZD = Z @ D
    meat0 = sum((Z[r] @ u[r]) ** 2 for r in rows_by.values())
    se_cl = np.sqrt((G / (G - 1)) * meat0 / (ZD ** 2))
    t_hat = beta_hat / se_cl
    p_cl = 2 * (1 - stats.t.cdf(abs(t_hat), df=G - 1))
    # Wild-cluster bootstrap with Webb weights, B=9999
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    rng = np.random.default_rng(42)
    B = 9999
    t_star = np.empty(B)
    for b in range(B):
        w_map = {c: webb[rng.integers(0, 6)] for c in clusters}
        wv = np.array([w_map[c] for c in g])
        Yb = D * 0.0 + Y * wv  # H0: beta=0
        bb = iv_beta(Yb, D, Z); ub = Yb - D * bb
        meatb = sum((Z[r] @ ub[r]) ** 2 for r in rows_by.values())
        seb = np.sqrt((G / (G - 1)) * meatb / (ZD ** 2))
        t_star[b] = bb / seb
