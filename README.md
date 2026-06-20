# Agricultural Climate Migration

County-level estimates of climate-driven yield loss, stranded farmland value, rural out-migration, crop insurance mispricing, and northern production opportunity. The analysis covers 2,902 US counties and eight field crops.

Author: Keshav Krishnan ([kkrishnan@parktudor.org](mailto:kkrishnan@parktudor.org))

## What is in this repo

| Area | Role |
|------|------|
| `src/` | Pipeline stages 1–10 (ingest through figures) |
| `src/revision/` | Extended stages 11–18 (DCF detail, IV migration, decomposition, robustness) |
| `results/revision/` | JSON outputs from the extended stages |
| `tests/` | `pytest`, 85% coverage gate |
| `data/raw/` | Download instructions (~12 GB inputs, not in git) |

## Requirements

Python 3.11, about 16 GB RAM, and 12 GB disk after downloading raw data. Conda is the easiest install path (`environment.yml`).

## Install

```bash
git clone https://github.com/keshavkrishnan08/ag-climate-migration.git
cd ag-climate-migration
conda env create -f environment.yml
conda activate agmigration
```

Pip alternative:

```bash
pip install numpy==1.26.4 pandas scipy lightgbm scikit-learn statsmodels linearmodels xarray netcdf4 geopandas
```

## Data

Fetch inputs once using [`data/raw/README.md`](data/raw/README.md). You need NASS yields and land values, PRISM climate, RMA insurance premiums, Census ACS and PEP files, CPI, ERS county typology, and CMIP6 projections. `make ingest` and `make features` write parquets under `data/processed/` (gitignored). Published county CSVs are described in [`data/published_dataset/README.md`](data/published_dataset/README.md).

## Pipeline

One analysis chain. Run everything with `make pipeline`, or run stages individually. `make pipeline-help` lists targets.

```bash
make pipeline    # all stages (~1 hr with cached data)
make verify      # rebuild HEADLINE_NUMBERS.json and spot-check values
make test        # pytest
```

### Stage 1 — Ingest (`make ingest`)

Loads raw NASS, PRISM, RMA, Census, and CPI files into normalized parquets under `data/processed/`. Script: `src/01_ingest.py`.

### Stage 2 — Features (`make features`)

Builds the county-year feature matrix: climate anomalies, soil proxies, lagged yields, crop shares. Script: `src/02_features.py`.

### Stage 3 — Yield model (`make model`)

Trains a per-crop LightGBM ensemble on historical yields with a strict temporal split (train ≤ 2009, validate 2010–2016, test 2017–2023). Script: `src/03_yield_model.py`.

### Stage 4 — Crop switching (`make switching`)

Estimates historical crop-switching rates from CDL land-cover transitions. Script: `src/04_switching.py`.

### Stage 5 — Projections (`make project`)

Applies CMIP6 SSP2-4.5 climate anomalies to the yield model through 2050. Script: `src/05_project.py`.

### Stage 6 — Stranded assets (`make stranded`)

First-pass discounted cash flow (DCF) valuation of farmland at risk from projected yield loss. Uses a 4% real discount rate and 30-year horizon. Script: `src/06_stranded.py`.

### Stage 7 — Cascade (`make cascade`)

Links farm-income shocks to prime-age out-migration and county population decline with a feedback loop. Script: `src/07_cascade.py`.

### Stage 8 — Insurance (`make insurance`)

Compares RMA premium rates based on frozen Actual Production History against rates that would apply under rolling APH. Script: `src/08_insurance.py`.

### Stage 9 — Northern frontier (`make frontier`)

Identifies counties where warming expands viable crop acreage and estimates net farm-income opportunity. Script: `src/09_frontier.py`.

### Stage 10 — Figures (`make figures`)

Generates 12 county-level maps and charts. Script: `src/10_figures.py`.

### Stage 11 — DCF and hedonic valuation (`make stranded-dcf`)

Refines stranded-farmland estimates with alternate-use floors ($1,500/ac pasture cap), soil- and irrigation-controlled hedonic regressions, propagated confidence intervals, and an ML-vs-process cross-check. Scripts in `src/revision/`: `stranded_revision.py`, `stranded_floor_sensitivity.py`, `hedonic_strengthened.py`, `dcf_ci_fixed.py`, `dollar_robustness.py`.

### Stage 12 — Insurance decomposition (`make insurance-decomp`)

Breaks gross mispricing into rolling-APH absorption, Trend-Adjusted Yield effects, Revenue Protection puts, Supplemental Coverage Option, and a reform-eliminable residual. Scripts: `insurance_rolling_aph.py`, `insurance_rp_and_tay.py`, `insurance_coverage_endogeneity.py`, `insurance_sco.py`.

### Stage 13 — Migration analysis (`make migration-analysis`)

Shift-share IV linking farm-income shocks to prime-age migration, wild-cluster bootstrap inference, instrument-balance checks, fiscal long-differences, and a depopulation cost Monte Carlo. Scripts: `migration_iv_bartik.py`, `migration_primeage_panel.py`, `migration_wildbootstrap.py`, `migration_share_balance.py`, `migration_fiscal_chain.py`, `migration_depop_montecarlo.py`, `migration_farmdependent.py`.

### Stage 14 — Yield model skill (`make yield-skill`)

Spectrum-based feature set with SSURGO water capacity and irrigation flags; decomposition of R² gains from target scaling vs feature engineering. Scripts: `yield_v7_spectrum.py`, `yield_audit_target_decomp.py`.

### Stage 15 — Framework tests (`make framework-tests`)

Tests whether forward warming predicts multiple economic channels jointly and runs a Granger-style cohesion check across stranded value, insurance, migration, and opportunity. Scripts: `framework_common_driver.py`, `framework_cohesion.py`.

### Stage 16 — Robustness batteries (`make robustness`)

Substantive falsification checks and tiered sensitivity grids (E1–E45). Scripts: `substantive_experiments.py`, `tier1_experiments.py` through `tier5_residuals.py`.

### Stage 17 — Adversarial checks (`make adversarial`)

Targeted falsification experiments (SEM partial-outs, DCF–hedonic CI overlap, IV leave-one-year-out, US-specific alternate-use floor, northern acreage expansion). Script: `robustness_battery.py`. Optional figure rebuild: `make figures-extra`.

### Stage 18 — Summary (`make summary`)

Merges key outputs into `results/revision/HEADLINE_NUMBERS.json`. Script: `headline_numbers.py`.

## Outputs

JSON files under `results/revision/` are committed. Parquet, CSV, and PDF files regenerate locally and stay gitignored. `make pipeline-clean` removes local JSON if you want a fresh run.

Output-to-script mapping: [`REPRODUCE.md`](REPRODUCE.md). Script index: [`src/revision/README.md`](src/revision/README.md).

## Conventions

2023 USD throughout (CPI base 304.7 from `data/raw/other/cpi_annual.csv`). Seed 42 for stochastic steps. FIPS codes are five-digit strings; aggregates 998 and 999 excluded.

## Citation and license

[`CITATION.cff`](CITATION.cff). MIT ([`LICENSE`](LICENSE)).
