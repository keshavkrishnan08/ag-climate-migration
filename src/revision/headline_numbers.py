"""Consolidate every headline number cited in the manuscript into one JSON.

Reads all per-experiment result JSONs in results/revision/ and writes a single
HEADLINE_NUMBERS.json that pairs each number cited in the paper with its source script,
input JSON, and computed value. Reviewers can regenerate everything with `make headline`
and grep this single file to verify any cited number against its provenance.

Seed 42; reads results/revision/*.json; writes results/revision/HEADLINE_NUMBERS.json.
"""
import json, glob, os
from pathlib import Path
OUT = Path("results/revision")

def jl(name):
    p = OUT / name
    return json.load(open(p)) if p.exists() else None

def get(d, *path, default=None):
    cur = d
    for k in path:
        if cur is None: return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur if cur is not None else default

# Load every per-experiment JSON
master = jl("MASTER_NUMBERS.json") or {}
ranged = jl("dcf_ci_fixed.json") or {}
floor = jl("stranded_floor_sensitivity.json") or {}
hedonic = jl("hedonic_strengthened.json") or {}
dollar = jl("dollar_robustness.json") or {}
ins_dec = jl("insurance_decomposition.json") or {}
ins_rp = jl("insurance_rp_tay.json") or {}
ins_cov = jl("insurance_coverage_endogeneity.json") or {}
ins_sco = jl("insurance_sco.json") or {}
mig_bar = jl("migration_iv_bartik.json") or {}
mig_pa = jl("migration_primeage_panel.json") or {}
mig_wcb = jl("migration_wildbootstrap.json") or {}
mig_sbl = jl("migration_share_balance.json") or {}
mig_dep = jl("migration_depop_montecarlo.json") or {}
mig_fc = jl("migration_fiscal_chain.json") or {}
mig_fd = jl("migration_farmdependent.json") or {}
mig_hi = jl("migration_high_tercile_2sls.json") or {}
mig_inf = jl("migration_inference_robust.json") or {}
yield_dec = jl("audit_yield_target_decomp.json") or {}
common = jl("framework_common_driver.json") or {}
substantive = jl("substantive_experiments.json") or {}
market = jl("market_robustness.json") or {}

HEADLINE = {
    "schema": ("Each entry maps a number CITED in the paper to its source script, input JSON, "
               "and computed value. Reviewers verify reproducibility by running the named script "
               "and checking the value against the JSON path."),

    # === STRANDED VALUE ===
    "stranded_conservative_DCF_B": {
        "value": 52,
        "source_script": "src/revision/stranded_revision.py",
        "source_json": "results/revision/stranded_central_floored.parquet (groupby fips, sum)",
        "computed_in": "substantive_experiments.json (cross-check)",
        "cited": "main.tex abstract, Results Table 1, SI S0"},
