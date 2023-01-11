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
