# Extended pipeline scripts

Stages 11–18 of the analysis. Invoked by `make stranded-dcf`, `insurance-decomp`, `migration-analysis`, `yield-skill`, `framework-tests`, `robustness`, `adversarial`, and `summary`. See the root README for what each stage computes.

## Active scripts

| Makefile target | Scripts | Output |
|-----------------|---------|--------|
| `stranded-dcf` | `stranded_revision.py`, `stranded_floor_sensitivity.py`, `hedonic_strengthened.py`, `dcf_ci_fixed.py`, `dollar_robustness.py` | JSON + local parquet |
| `insurance-decomp` | `insurance_rolling_aph.py`, `insurance_rp_and_tay.py`, `insurance_coverage_endogeneity.py`, `insurance_sco.py` | `insurance_*.json` |
| `migration-analysis` | `migration_farmdependent.py`, `migration_iv_bartik.py`, `migration_primeage_panel.py`, `migration_wildbootstrap.py`, `migration_share_balance.py`, `migration_fiscal_chain.py`, `migration_depop_montecarlo.py` | `migration_*.json` |
| `yield-skill` | `yield_v7_spectrum.py`, `yield_audit_target_decomp.py` | `yield_v7_metrics.json`, `audit_yield_target_decomp.json` |
| `framework-tests` | `framework_cohesion.py`, `framework_common_driver.py` | `framework_*.json` |
| `robustness` | `substantive_experiments.py`, `tier1_experiments.py` … `tier5_residuals.py` | `substantive_experiments.json`, `tier*.json` |
| `adversarial` | `robustness_battery.py` | `results/revision/adversarial/*.json` |
| `figures-extra` | `adversarial_figures.py`, `si_graphics.py` | PDFs in `results/figures_revision/` |
| `summary` | `headline_numbers.py` | `HEADLINE_NUMBERS.json` |

## Utilities

`insurance_fast.py`, `pull_ssurgo.py`, `fig07_marginal.py`, `recompute_opportunity.py`, `dcf_ge_price_sensitivity.py`

## Archive

Older ablation scripts (`migration_iv_v2.py`, `yield_v4_morefeatures.py`, `opportunity_clean.py`, etc.) stay in the tree for audit. They are not part of `make pipeline`.

See [`../../REPRODUCE.md`](../../REPRODUCE.md) for output tables.
