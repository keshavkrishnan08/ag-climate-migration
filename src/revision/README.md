# `src/revision/` scripts

Experiment scripts for secondary analyses. The root `Makefile` runs the **active** set via `make reproduce`.

## Active scripts (in `make reproduce`)

| Script | Output JSON |
|--------|-------------|
| `stranded_revision.py` | parquet + downstream JSONs |
| `stranded_floor_sensitivity.py` | `stranded_floor_sensitivity.json` |
| `hedonic_strengthened.py` | `hedonic_strengthened.json` |
| `dcf_ci_fixed.py` | `dcf_ci_fixed.json` |
| `dollar_robustness.py` | `dollar_robustness.json` |
| `insurance_rolling_aph.py` | `insurance_decomposition.json` |
| `insurance_rp_and_tay.py` | `insurance_rp_tay.json` |
| `insurance_coverage_endogeneity.py` | `insurance_coverage_endogeneity.json` |
| `insurance_sco.py` | `insurance_sco.json` |
| `migration_farmdependent.py` | `migration_farmdependent.json` |
| `migration_iv_bartik.py` | `migration_iv_bartik.json` (+ related) |
| `migration_primeage_panel.py` | `migration_primeage_panel.json` |
| `migration_wildbootstrap.py` | `migration_wildbootstrap.json` |
| `migration_share_balance.py` | `migration_share_balance.json` |
| `migration_fiscal_chain.py` | `migration_fiscal_chain.json` |
| `migration_depop_montecarlo.py` | `migration_depop_montecarlo.json` |
| `recompute_opportunity.py` | CSV (local) |
| `yield_v7_spectrum.py` | `yield_v7_metrics.json` |
| `yield_audit_target_decomp.py` | `audit_yield_target_decomp.json` |
| `framework_common_driver.py` | `framework_common_driver.json` |
| `framework_cohesion.py` | `framework_cohesion.json` |
| `substantive_experiments.py` | `substantive_experiments.json` |
| `tier1_experiments.py` … `tier5_residuals.py` | matching `tier*.json` |
| `headline_numbers.py` | `HEADLINE_NUMBERS.json` |
| `robustness_battery.py` | `adversarial/e55_*.json` … `e64_*.json` |
| `adversarial_figures.py` | PDFs in `results/figures_revision/` (local) |
| `si_graphics.py` | PDFs in `results/figures_revision/` (local) |

## Adversarial battery (`make rev-adversarial`)

| Script | Output |
|--------|--------|
| `robustness_battery.py` | `results/revision/adversarial/*.json` |

## Utilities

| Script | Purpose |
|--------|---------|
| `insurance_fast.py` | Faster insurance grid runs |
| `pull_ssurgo.py` | SSURGO download |
| `fig07_marginal.py` | Fig 7B marginal effects |

## Superseded (not in `make reproduce`)

Kept for audit trail. See script headers or git history for replacements.

`migration_iv_v2.py`, `migration_multiiv.py`, `migration_primeage.py`, `migration_longdiff.py`, `migration_robustness.py`, `opportunity_clean.py`, `yield_v4_morefeatures.py`, `yield_v5_percrop.py`, `yield_v6_skill.py`, `yield_v7_baseline.py`, `yield_model_v3_features.py`, `yield_audit_*.py` (except `target_decomp`), `yield_irr_final.py`, `yield_levels_direct.py`, `yield_monotonic.py`, `yield_stack_levels.py`, `yield_uncertainty.py`

## Run

```bash
make reproduce
make headline
make verify

make rev-stranded    # single subsystem
```

Inputs: `data/raw/`, `data/processed/`. Outputs: `results/revision/*.json`.

See `../../REPRODUCE.md` for output tables.
