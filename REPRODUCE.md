# Reproducibility Guide — Every Headline Number

This document maps every number cited in the revised manuscript to its source script and result JSON. Reviewers can verify any cited value by running the named script and checking the JSON path.

## Quick start

```bash
# 1. Set up environment (Python 3.11 + numpy 1.26.4, lightgbm, scipy, pandas)
pip install --break-system-packages numpy==1.26.4 pandas scipy lightgbm scikit-learn

# 2. Reproduce every headline number used in the paper
make reproduce

# 3. Consolidate into a single auditable file
make headline   # writes results/revision/HEADLINE_NUMBERS.json
```

The `HEADLINE_NUMBERS.json` is the single source-of-truth file: 35 cited values, 20 auto-verified against per-script JSONs (the rest are computed-then-rounded for the manuscript).

## Headline numbers → source scripts

### Stranded farmland value ($52–80B field-crop; $168B all-channel upper bound)

| Number | Script | Output |
|---|---|---|
| Conservative DCF $52B | `src/revision/stranded_revision.py` | `results/revision/stranded_central_floored.parquet` |
| Central floored $61B | `src/revision/stranded_revision.py` | (same; $1,500/ac floor) |
| Soil-controlled hedonic $80B | `src/revision/hedonic_strengthened.py` | `hedonic_strengthened.json` |
| Propagated CI [$37, $77]B | `src/revision/dcf_ci_fixed.py` | `dcf_ci_fixed.json` |
| Floor sensitivity ($1k/$2k) → $68B/$52B | `src/revision/stranded_floor_sensitivity.py` | `stranded_floor_sensitivity.json` |
| ML vs process ($59B vs $13B, ρ=0.66) | `src/revision/dollar_robustness.py` | `dollar_robustness.json` |
| All-channel upper bound $168/$183B (DCF-scaling) | `src/revision/hedonic_strengthened.py` | `hedonic_strengthened.json` |

### Insurance ($6.6B gross → $3.7B residual; $1.6B transfer)

| Number | Script | Output |
|---|---|---|
| Decomposition $6.6 → −2.0 → −0.9 → $3.7B | `src/revision/insurance_rolling_aph.py` | `insurance_decomposition.json` |
| RP residual $2.6B (dominant product) | `src/revision/insurance_rp_and_tay.py` | `insurance_rp_tay.json` |
| Acreage-weighted coverage 0.74 | `src/revision/insurance_coverage_endogeneity.py` | `insurance_coverage_endogeneity.json` |
| TAY participation sensitivity | `src/revision/insurance_rp_and_tay.py` | `insurance_rp_tay.json` |
| SCO contribution +$0.01B | `src/revision/insurance_sco.py` | `insurance_sco.json` |
| Process-based falsification $0.83B | `src/revision/substantive_experiments.py` (E1) | `substantive_experiments.json` |
| Climate-σ sensitivity $3.92B | `src/revision/substantive_experiments.py` (E2) | `substantive_experiments.json` |

### Migration / rural decline (β=0.024 3-yr; 0.049 5-yr; depopulation $18B)

| Number | Script | Output |
|---|---|---|
| Leave-one-out shift-share panel | `src/revision/migration_iv_bartik.py` | `migration_iv_bartik.json` |
| Prime-age FE (3-yr, β=0.024, p=0.005, F=78, 429 cty) | `src/revision/migration_primeage_panel.py` | `migration_primeage_panel.json` |
| 5-yr horizon + county-clustered p=0.001; wild-cluster bootstrap p=0.0005 | `src/revision/migration_wildbootstrap.py` | `migration_wildbootstrap.json` |
| Two-way p=0.11; non-overlap β=0.059, p=0.012 | `src/revision/migration_iv_bartik.py` | `migration_inference_robust.json` |
| Total-pop tercile (β=0.053, p=0.004, F=94, 750 cty) | `src/revision/migration_iv_bartik.py` | `migration_high_tercile_2sls.json` |
| Goldsmith-Pinkham / Borusyak-Hull-Jaravel balance | `src/revision/migration_share_balance.py` | `migration_share_balance.json` |
| Non-farm effect-size dominance (63×) | `src/revision/substantive_experiments.py` (E3–E4) | `substantive_experiments.json` |
| Depopulation Monte Carlo ($18B central, $22B median, [$11, $38]B) | `src/revision/migration_depop_montecarlo.py` | `migration_depop_montecarlo.json` |
| National welfare floor (frictional, $4.3B) | `src/revision/substantive_experiments.py` (E6) | `substantive_experiments.json` |
| Fiscal chain (long-difference revenue→land value) | `src/revision/migration_fiscal_chain.py` | `migration_fiscal_chain.json` → `revenue_to_landvalue_longdiff` |

### Northern opportunity ($8.1B net; $37B gross; 514 counties)

| Number | Script | Output |
|---|---|---|
| Net farm income $8.1B / gross $37B | `src/revision/recompute_opportunity.py` | (per-county CSV) |
| Per-state breakdown | `src/revision/recompute_opportunity.py` | |

### Yield model (R²=0.41 anomaly; 0.68 levels; 0.75 spatial)

| Number | Script | Output |
|---|---|---|
| Spectrum on %-deviation R²=0.41 | `src/revision/yield_v7_spectrum.py` | `yield_v7_metrics.json` |
| z-anomaly apples-to-apples (features +0.05; target shift +0.29) | `src/revision/yield_audit_target_decomp.py` | `audit_yield_target_decomp.json` |
| SSURGO pull | `src/revision/pull_ssurgo.py` | (SSURGO parquet) |

### Common-cause / framework

| Number | Script | Output |
|---|---|---|
| Forward warming predicts 3/4 channels | `src/revision/framework_common_driver.py` | `framework_common_driver.json` |
