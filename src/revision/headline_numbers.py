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
    "stranded_central_floored_DCF_B": {
        "value": 61,
        "source_script": "src/revision/stranded_revision.py",
        "source_json": "results/revision/stranded_central_floored.parquet (1500/ac floor)",
        "value_recomputed": get(floor, "pasture_1500_per_ac_central_B"),
        "cited": "main.tex Results §Stranded, SI S0"},
    "stranded_floor_sensitivity_B": {
        "values": {"pasture_1000": get(floor, "pasture_1000_per_ac_central_B"),
                   "pasture_1500": get(floor, "pasture_1500_per_ac_central_B"),
                   "pasture_2000": get(floor, "pasture_2000_per_ac_central_B")},
        "source_script": "src/revision/stranded_floor_sensitivity.py",
        "source_json": "results/revision/stranded_floor_sensitivity.json",
        "cited": "main.tex Results, response Major 1 table"},
    "hedonic_soil_irrigation_controlled_B": {
        "value": 80,
        "value_recomputed": get(hedonic, "specs", "+ SSURGO + irrigation + soil-productivity", "stranded_B"),
        "marginal_pct_per_F": get(hedonic, "specs", "+ SSURGO + irrigation + soil-productivity", "marginal_pct_per_F_at_mean"),
        "coefficient_stability_pct": get(hedonic, "coef_stability_pct"),
        "source_script": "src/revision/hedonic_strengthened.py",
        "source_json": "results/revision/hedonic_strengthened.json",
        "cited": "main.tex Results §Stranded, SI S12, S14"},
    "all_channel_upper_bound_B": {
        "value": 168,
        "DCF_scaled_route": get(hedonic, "implied_all_channel_from_dcf_B"),
        "source_script": "src/revision/hedonic_strengthened.py",
        "source_json": "results/revision/hedonic_strengthened.json",
        "cited": "main.tex abstract, Results §Stranded, SI S12"},
    "stranded_propagated_CI_B": {
        "value": [37, 77],
        "value_recomputed": get(ranged, "propagated_full_CI_B"),
        "source_script": "src/revision/dcf_ci_fixed.py",
        "source_json": "results/revision/dcf_ci_fixed.json",
        "cited": "main.tex Methods, Results, SI S10"},
    "stranded_ML_vs_process_B": {
        "ML_path": get(dollar, "national_total_ml_B"),
        "process_path": get(dollar, "national_total_process_B"),
        "spatial_rank_rho": get(dollar, "spatial_rank_correlation"),
        "source_script": "src/revision/dollar_robustness.py",
        "source_json": "results/revision/dollar_robustness.json",
        "cited": "SI §Substantive E8"},

    # === INSURANCE ===
    "insurance_gross_frozen_B": {
        "value": 6.6, "value_recomputed": get(ins_dec, "gross_frozen_total_B"),
        "source_script": "src/revision/insurance_rolling_aph.py",
        "source_json": "results/revision/insurance_decomposition.json"},
    "insurance_rolling_total_B": {
        "value": 4.6, "value_recomputed": get(ins_dec, "rolling_total_B"),
        "source_script": "src/revision/insurance_rolling_aph.py",
        "source_json": "results/revision/insurance_decomposition.json"},
    "insurance_residual_TAY_total_B": {
        "value": 3.7, "value_recomputed": get(ins_dec, "residual_tay_total_B"),
        "source_script": "src/revision/insurance_rolling_aph.py",
        "source_json": "results/revision/insurance_decomposition.json",
        "cited": "main.tex abstract, Results §4.3, Table insurance_flows"},
    "insurance_transfer_B": {
        "value": 1.6, "value_recomputed": get(ins_dec, "residual_tay_xsub_B"),
        "source_script": "src/revision/insurance_rolling_aph.py",
        "source_json": "results/revision/insurance_decomposition.json"},
    "insurance_RP_residual_B": {
