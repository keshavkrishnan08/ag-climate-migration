# Replication package — output reference

Maps pipeline outputs to source scripts and JSON paths. For setup and workflow, start with [`README.md`](README.md).

## Quick start

```bash
conda activate agmigration
make pipeline
make verify    # rebuilds and checks results/revision/HEADLINE_NUMBERS.json
```

JSON summaries are committed under `results/revision/`. Parquet and CSV files are local only.

## Stranded farmland value

| Output | Script | JSON |
|--------|--------|------|
| Conservative DCF $52B | `stranded_revision.py` | parquet (local) |
| Central floored $61B | `stranded_revision.py` | parquet (local) |
| Soil-controlled hedonic $80B | `hedonic_strengthened.py` | `hedonic_strengthened.json` |
| Propagated CI [$37, $77]B | `dcf_ci_fixed.py` | `dcf_ci_fixed.json` |
| Floor sensitivity | `stranded_floor_sensitivity.py` | `stranded_floor_sensitivity.json` |
| ML vs process | `dollar_robustness.py` | `dollar_robustness.json` |
| All-channel upper bound | `hedonic_strengthened.py` | `hedonic_strengthened.json` |

## Insurance

| Output | Script | JSON |
|--------|--------|------|
| Decomposition chain | `insurance_rolling_aph.py` | `insurance_decomposition.json` |
| RP residual | `insurance_rp_and_tay.py` | `insurance_rp_tay.json` |
| Coverage 0.74 | `insurance_coverage_endogeneity.py` | `insurance_coverage_endogeneity.json` |
| TAY sensitivity | `insurance_rp_and_tay.py` | `insurance_rp_tay.json` |
| SCO | `insurance_sco.py` | `insurance_sco.json` |
| Process falsification (E1) | `substantive_experiments.py` | `substantive_experiments.json` |
| Climate-σ sensitivity (E2) | `substantive_experiments.py` | `substantive_experiments.json` |

## Migration

| Output | Script | JSON |
|--------|--------|------|
| Shift-share IV | `migration_iv_bartik.py` | `migration_iv_bartik.json` |
| Prime-age panel FE | `migration_primeage_panel.py` | `migration_primeage_panel.json` |
| Wild-cluster bootstrap | `migration_wildbootstrap.py` | `migration_wildbootstrap.json` |
| Inference robustness | `migration_iv_bartik.py` | `migration_inference_robust.json` |
| High-tercile 2SLS | `migration_iv_bartik.py` | `migration_high_tercile_2sls.json` |
| Share balance | `migration_share_balance.py` | `migration_share_balance.json` |
| Non-farm dominance (E3–E4) | `substantive_experiments.py` | `substantive_experiments.json` |
| Depopulation MC | `migration_depop_montecarlo.py` | `migration_depop_montecarlo.json` |
| Welfare floor (E6) | `substantive_experiments.py` | `substantive_experiments.json` |
| Fiscal chain | `migration_fiscal_chain.py` | `migration_fiscal_chain.json` |

## Northern opportunity

| Output | Script | JSON / file |
|--------|--------|-------------|
| Net / gross farm income | `recompute_opportunity.py` | CSV (local) |
| Per-state breakdown | `recompute_opportunity.py` | CSV (local) |

## Yield model

| Output | Script | JSON |
|--------|--------|------|
| Spectrum R²=0.41 | `yield_v7_spectrum.py` | `yield_v7_metrics.json` |
| Target decomposition | `yield_audit_target_decomp.py` | `audit_yield_target_decomp.json` |
| SSURGO pull | `pull_ssurgo.py` | parquet (local) |

## Framework

| Output | Script | JSON |
|--------|--------|------|
| Common-cause test | `framework_common_driver.py` | `framework_common_driver.json` |
| Chain test | `framework_cohesion.py` | `framework_cohesion.json` |

## Experiment batteries

| Script | JSON |
|--------|------|
| `substantive_experiments.py` | `substantive_experiments.json` |
| `tier1_experiments.py` | `tier1_experiments.json` |
| `tier2_experiments.py` | `tier2_experiments.json` |
| `tier3_tighten.py` | `tier3_tighten.json` |
| `tier4_refit.py` | `tier4_refit.json` |
| `tier5_residuals.py` | `tier5_residuals.json` |
| `robustness_battery.py` | `supplementary/e55_sem_partialouts.json` … `e64_northern_acreage.json` |

All scripts live under `src/revision/`.

## Raw data

Not in git. See `data/raw/README.md`.

| Source | Path |
|--------|------|
| NASS yields | `data/raw/nass/nass_county_yields.parquet` |
| PRISM climate | `data/raw/prism/county_climate_*.parquet` |
| RMA SOB | `data/raw/rma/rma_sob_all_years.parquet` |
| Census ACS | `data/raw/census/acs_*.parquet` |
| CMIP6 | `data/projections/*.parquet` |
| ERS Atlas | `data/raw/other/ers_atlas/*.csv` |
| Census PEP | `data/raw/census/cc-est*.csv` |
| CPI | `data/raw/other/cpi_annual.csv` |
| SSURGO | pulled via `pull_ssurgo.py` |

## Conventions

- 2023 USD (CPI_2023 = 304.7)
- Seed 42 for stochastic steps
- FIPS zero-padded; 998/999 filtered
- ML split: train ≤ 2009, validate 2010–2016, test 2017–2023

Active vs superseded scripts: `src/revision/README.md`.
