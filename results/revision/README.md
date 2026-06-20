# Revision experiment outputs

JSON summaries from `src/revision/` scripts. Parquet and CSV artifacts are gitignored (regenerate locally).

## Key files

| File | Role |
|------|------|
| `HEADLINE_NUMBERS.json` | Index of primary outputs → source script |
| `MASTER_NUMBERS.json` | Extended cross-check table |
| `substantive_experiments.json` | E1–E9 battery |
| `tier1_experiments.json` … `tier5_residuals.json` | Tier batteries |
| `adversarial/e55_*.json` … `e64_*.json` | Adversarial robustness battery |

## Regenerate

```bash
make reproduce
make headline
make verify
```

Requires raw data (~12 GB). See `data/raw/README.md`.

## Groups

**Stranded:** `hedonic_strengthened.json`, `dcf_ci_fixed.json`, `dollar_robustness.json`, `stranded_floor_sensitivity.json`

**Insurance:** `insurance_decomposition.json`, `insurance_rp_tay.json`, `insurance_coverage_endogeneity.json`, `insurance_sco.json`

**Migration:** `migration_iv_bartik.json`, `migration_primeage_panel.json`, `migration_wildbootstrap.json`, `migration_depop_montecarlo.json`, …

**Yield:** `yield_v7_metrics.json`, `audit_yield_target_decomp.json`, …

**Framework:** `framework_common_driver.json`, `framework_cohesion.json`

Other JSON files are robustness or superseded runs retained for transparency.
