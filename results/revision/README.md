# Pipeline JSON outputs

Machine-readable results from stages 11–18. Parquet and CSV side files are local only.

## Key files

| File | Stage |
|------|-------|
| `HEADLINE_NUMBERS.json` | Summary (stage 18) |
| `MASTER_NUMBERS.json` | Extended cross-check table |
| `hedonic_strengthened.json`, `dcf_ci_fixed.json`, … | DCF / hedonic (stage 11) |
| `insurance_decomposition.json`, … | Insurance decomposition (stage 12) |
| `migration_iv_bartik.json`, … | Migration IV (stage 13) |
| `yield_v7_metrics.json`, … | Yield skill (stage 14) |
| `framework_common_driver.json`, … | Framework tests (stage 15) |
| `substantive_experiments.json`, `tier*.json` | Robustness (stage 16) |
| `adversarial/e55_*.json`, … | Adversarial (stage 17) |

## Regenerate

```bash
make pipeline
make verify
```

Requires raw data. See [`../../data/raw/README.md`](../../data/raw/README.md).
