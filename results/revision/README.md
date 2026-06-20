# Pipeline JSON outputs

Machine-readable results from the extended analysis modules (Steps 11–17). Parquet and CSV side files are local only.

## Key files

| File | Step |
|------|------|
| `HEADLINE_NUMBERS.json` | Summary (Step 17) |
| `MASTER_NUMBERS.json` | Extended cross-check table |
| `hedonic_strengthened.json`, `dcf_ci_fixed.json`, … | DCF / hedonic (Step 11) |
| `insurance_decomposition.json`, … | Insurance decomposition (Step 12) |
| `migration_iv_bartik.json`, … | Migration IV (Step 13) |
| `yield_v7_metrics.json`, … | Yield model skill (Step 14) |
| `framework_common_driver.json`, … | Framework tests (Step 15) |
| `substantive_experiments.json`, `tier*.json` | Robustness grids (Step 16) |
| `supplementary/e55_*.json`, … | Supplementary robustness E55–E64 (Step 16) |

## Regenerate

```bash
make pipeline
make verify
```

Requires raw data. See [`../../data/raw/README.md`](../../data/raw/README.md).
