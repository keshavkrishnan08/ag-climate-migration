# Agricultural Climate Migration

County-level estimates of climate-driven yield loss, stranded farmland value, rural out-migration, crop insurance mispricing, and northern production opportunity. The analysis covers 2,902 US counties and eight field crops.

Author: Keshav Krishnan ([kkrishnan@parktudor.org](mailto:kkrishnan@parktudor.org))

## What is in this repo

| Area | Files | Role |
|------|-------|------|
| `src/` | 49 Python modules | Ingest, features, yield model, switching, projections, stranded assets, cascade, insurance, frontier, figures |
| `src/revision/` | 63 scripts | Secondary experiments and sensitivity checks |
| `results/revision/` | 85 JSON files | Stored outputs (5 more under `adversarial/`) |
| `tests/` | unit + integration | `pytest`, 85% coverage gate |
| `data/raw/` | README only in git | Download instructions for ~12 GB of inputs |

Paper and submission PDFs stay outside this repository.

## Requirements

Python 3.11, about 16 GB RAM for the full county panel, and 12 GB free disk once raw data are downloaded. Conda is the easiest path (`environment.yml`).

## Install

```bash
git clone https://github.com/keshavkrishnan08/ag-climate-migration.git
cd ag-climate-migration
conda env create -f environment.yml
conda activate agmigration
```

Pip-only install if you prefer it:

```bash
pip install numpy==1.26.4 pandas scipy lightgbm scikit-learn statsmodels linearmodels xarray netcdf4 geopandas
```

## Data

Nothing under `data/raw/` ships with the clone. Fetch inputs once using [`data/raw/README.md`](data/raw/README.md). Expect roughly three hours on a decent connection.

You need NASS county yields and land values, PRISM monthly climate, RMA Summary of Business premiums, Census ACS and PEP population files, a CPI series, ERS county typology codes, and CMIP6 SSP projections. `make ingest` and `make features` then write parquets under `data/processed/`, which are also gitignored.

Published county CSVs and the full dataset tarball are documented in [`data/published_dataset/README.md`](data/published_dataset/README.md).

## Main pipeline

Runs in order from the repo root:

```bash
make ingest      # raw → processed parquets
make features    # county feature matrix
make model       # LightGBM yield ensemble
make switching   # crop switching rates
make project     # CMIP6 yield projections
make stranded    # DCF stranded farmland value
make cascade     # rural decline feedback
make insurance   # RMA mispricing
make frontier    # northern opportunity counties
make figures     # 12 publication figures
make test        # pytest
```

Or `make all` for the full chain. Stage scripts live in `src/` (`01_ingest.py` through `10_figures.py`). Supporting modules handle CMIP6 downloads, hedonic decompositions, uncertainty propagation, and SSP3-7.0 reruns (`run_projections_ssp370.py`, `run_stranded_ssp370.py`).

## Revision experiments

The `src/revision/` folder holds follow-on work: IV migration specs, insurance decomposition, hedonic cross-checks, yield model audits, tier batteries, and adversarial falsification tests. Most write JSON under `results/revision/`.

```bash
make reproduce        # full active set (~45 min with cached data)
make rev-adversarial  # E55, E56, E58, E60, E64 battery
make rev-figures      # PDFs from JSONs → results/figures_revision/ (local)
make headline         # merge key numbers → HEADLINE_NUMBERS.json
make verify           # print stored vs recomputed values
```

Run one block at a time with `make rev-stranded`, `rev-insurance`, `rev-migration`, `rev-yield`, `rev-framework`, or `rev-substantive`. `make rev-help` lists targets.

### Script groups (63 files)

**Stranded and hedonic** — `stranded_revision.py`, `stranded_floor_sensitivity.py`, `hedonic_strengthened.py`, `dcf_ci_fixed.py`, `dollar_robustness.py`, `dcf_ge_price_sensitivity.py`

**Insurance** — `insurance_rolling_aph.py`, `insurance_rp_and_tay.py`, `insurance_coverage_endogeneity.py`, `insurance_sco.py`, `insurance_fast.py`

**Migration** — `migration_iv_bartik.py`, `migration_primeage_panel.py`, `migration_wildbootstrap.py`, `migration_share_balance.py`, `migration_fiscal_chain.py`, `migration_depop_montecarlo.py`, `migration_farmdependent.py`, plus earlier IV variants kept for audit (`migration_iv_v2.py`, `migration_multiiv.py`, `migration_robustness.py`, …)

**Yield** — `yield_v7_spectrum.py`, `yield_audit_target_decomp.py`, and a dozen audit or ablation scripts (`yield_v4_morefeatures.py`, `yield_monotonic.py`, `yield_spatial_block_perturbation.py`, …)

**Opportunity** — `recompute_opportunity.py`, `opportunity_clean.py`, `opportunity_soil_adjusted.py`, `pull_ssurgo.py`

**Framework** — `framework_common_driver.py`, `framework_cohesion.py`, `common_cause_sem.py`

**Experiment batteries** — `substantive_experiments.py`, `tier1_experiments.py` through `tier5_residuals.py`, `base_improvements.py`

**Adversarial** — `robustness_battery.py`, `adversarial_figures.py`, `si_graphics.py`

**Utilities** — `headline_numbers.py`, `fig07_marginal.py`

Superseded scripts remain in the tree so you can see what was tried. They are listed in [`src/revision/README.md`](src/revision/README.md). [`REPRODUCE.md`](REPRODUCE.md) maps each headline output to its script and JSON path.

## Outputs

JSON summaries in `results/revision/` are committed. Open `results/revision/HEADLINE_NUMBERS.json` without rerunning anything.

Parquet, CSV, and PDF artifacts are regenerated locally and gitignored. Adversarial JSONs live in `results/revision/adversarial/`. Figure PDFs from `make rev-figures` go to `results/figures_revision/`.

`make revision-clean` deletes local JSON outputs if you want a fresh run.

## Tests

```bash
make test
# or: pytest tests/ --cov=src --cov-fail-under=85
```

## Conventions

All dollar figures are 2023 USD (CPI from `data/raw/other/cpi_annual.csv`, base index 304.7). Stochastic steps use seed 42. County FIPS codes are five-digit strings; state aggregates 998 and 999 are dropped. The yield model trains on years through 2009, validates 2010–2016, and tests 2017–2023 with a two-year gap.

## Citation and license

[`CITATION.cff`](CITATION.cff) for software metadata. Code is MIT ([`LICENSE`](LICENSE)).
