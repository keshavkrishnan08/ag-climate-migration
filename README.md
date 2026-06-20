# Agricultural Climate Migration — replication code

Code to reproduce county-level estimates of climate-driven yield loss, stranded farmland value, rural out-migration, crop insurance mispricing, and northern production opportunity for 2,902 US counties and eight field crops.

**Author:** Keshav Krishnan ([kkrishnan@parktudor.org](mailto:kkrishnan@parktudor.org))

---

## Code availability

| Item | Detail |
|------|--------|
| **What** | Python pipeline for data ingest, yield projection, DCF valuation, migration IV, insurance decomposition, and robustness checks |
| **Where** | https://github.com/keshavkrishnan08/ag-climate-migration |
| **Version** | v1.0.0 (see [`CITATION.cff`](CITATION.cff) and Git tags) |
| **License** | MIT ([`LICENSE`](LICENSE)) |
| **Archive** | Mint a Zenodo DOI from the release tag used in the manuscript; cite that DOI in the paper reference list |
| **Restrictions** | Raw third-party data are not redistributed; download instructions are in [`data/raw/README.md`](data/raw/README.md) |

---

## Computational requirements

| Component | Specification |
|-----------|---------------|
| **OS** | macOS or Linux (tested); Windows via WSL should work |
| **Python** | 3.11 |
| **Memory** | ~16 GB RAM |
| **Disk** | ~12 GB after raw data download |
| **Environment** | [`environment.yml`](environment.yml) (Conda recommended) |
| **Random seed** | 42 for all stochastic steps |

Typical runtime: about one hour for `make pipeline` with processed parquets already on disk; first run is longer after downloading raw inputs.

---

## Quick start

```bash
git clone https://github.com/keshavkrishnan08/ag-climate-migration.git
cd ag-climate-migration
conda env create -f environment.yml
conda activate agmigration
```

1. Download raw inputs following [`data/raw/README.md`](data/raw/README.md).
2. Run the full analysis: `make pipeline`
3. Check headline numbers: `make verify`
4. Run unit tests: `make test`

Individual stages: `make pipeline-help`

---

## Repository layout

| Path | Contents |
|------|----------|
| `src/` | Primary pipeline scripts (`01_ingest.py` … `10_figures.py`) |
| `src/revision/` | Extended econometric modules (valuation, IV, decomposition, robustness) |
| `results/revision/` | Committed JSON summaries (git-tracked) |
| `results/revision/supplementary/` | Supplementary robustness outputs (E55–E64) |
| `data/raw/` | Data acquisition guide (files not in git) |
| `data/processed/` | Intermediate parquets (local, gitignored) |
| `tests/` | Pytest suite (85% coverage gate) |
| [`REPRODUCE.md`](REPRODUCE.md) | Table-level map from scripts to outputs |

Parquet, CSV, and figure PDFs regenerate locally and are not committed.

---

## Reproduction workflow

Run `make pipeline` to execute all steps in order, or invoke Makefile targets individually.

### Data and projections (Steps 1–5)

| Step | Target | Script | Description |
|------|--------|--------|-------------|
| 1 | `ingest` | `01_ingest.py` | Load NASS, PRISM, RMA, Census, CPI into `data/processed/` |
| 2 | `features` | `02_features.py` | County-year feature matrix (climate, soil, lags, crop shares) |
| 3 | `model` | `03_yield_model.py` | Per-crop LightGBM; train ≤2009, validate 2010–2016, test 2017–2023 |
| 4 | `switching` | `04_switching.py` | Historical crop-switching rates from CDL |
| 5 | `project` | `05_project.py` | CMIP6 SSP2-4.5 yield projections through 2050 |

### Core economic modules (Steps 6–10)

| Step | Target | Script | Description |
|------|--------|--------|-------------|
| 6 | `stranded` | `06_stranded.py` | DCF valuation of farmland at risk (4% real rate, 30-year horizon) |
| 7 | `cascade` | `07_cascade.py` | Farm-income shocks → prime-age migration → population feedback |
| 8 | `insurance` | `08_insurance.py` | Frozen vs rolling Actual Production History premium comparison |
| 9 | `frontier` | `09_frontier.py` | Northern counties with warming-driven production opportunity |
| 10 | `figures` | `10_figures.py` | Main county maps and charts |

### Extended analysis (Steps 11–16)

| Step | Target | Key outputs |
|------|--------|-------------|
| 11 | `stranded-dcf` | Hedonic regressions, alternate-use floors, propagated DCF confidence intervals |
| 12 | `insurance-decomp` | Rolling-APH, TAY, Revenue Protection, SCO, reform-eliminable residual |
| 13 | `migration-analysis` | Shift-share IV, wild-cluster bootstrap, fiscal long-differences, depopulation MC |
| 14 | `yield-skill` | SSURGO-augmented yield model and R² decomposition |
| 15 | `framework-tests` | Common-driver and cross-channel cohesion tests |
| 16 | `robustness` | Sensitivity grids (E1–E45) and supplementary checks (E55–E64) via `robustness_battery.py` |

Optional: `make figures-extra` rebuilds extended PDFs from JSON (`supplementary_figures.py`, `si_graphics.py`).

### Summary (Step 17)

| Step | Target | Output |
|------|--------|--------|
| 17 | `summary` | `results/revision/HEADLINE_NUMBERS.json` |

`make verify` rebuilds the summary file and prints stored vs recomputed headline values.

---

## Data availability

Raw data are **not** included in this repository. Sources: USDA NASS, PRISM, RMA, Census ACS/PEP, BLS CPI, ERS county typology, CMIP6 projections. Step-by-step download paths and file names: [`data/raw/README.md`](data/raw/README.md).

Published county-level CSVs (where applicable): [`data/published_dataset/README.md`](data/published_dataset/README.md).

---

## Conventions

- Dollar values in **2023 USD** (CPI base 304.7 from `data/raw/other/cpi_annual.csv`)
- FIPS codes as five-digit strings; state aggregates 998 and 999 excluded
- Random seed **42** for bootstrap and Monte Carlo draws

---

## Citation

If you use this code, cite the software metadata in [`CITATION.cff`](CITATION.cff). After archiving on Zenodo, cite the DOI in your reference list as recommended by [Springer Nature code-sharing guidance](https://www.springernature.com/gp/open-science/code-policy).

Questions: kkrishnan@parktudor.org
