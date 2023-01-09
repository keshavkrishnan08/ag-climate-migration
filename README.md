# Agricultural Climate Migration — US County Economic Costs

Code and manuscript for **The Economic Cost of Agricultural Climate Migration in the United States**.

**Author:** Keshav Krishnan ([kkrishnan@parktudor.org](mailto:kkrishnan@parktudor.org))

## Overview

County-level analysis (2,902 counties, eight field crops) linking climate-driven yield loss to:

1. **Stranded farmland value** ($52–80B field-crop; hedonic cross-check)
2. **Rural community decline** (farm-income IV → prime-age migration)
3. **Federal crop insurance mispricing** ($2.6–3.7B/yr reform-eliminable residual)
4. **Northern production opportunity** ($8.1B/yr net farm income uncaptured)

## Quick start

```bash
conda env create -f environment.yml
conda activate agmigration

# Original pipeline (figures + initial manuscript)
make all

# Revision headline numbers (Communications Sustainability resubmission)
make reproduce
make headline    # → results/revision/HEADLINE_NUMBERS.json
make verify
```

See [REPRODUCE.md](REPRODUCE.md) for the full map from every cited number to its source script.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/` | Original end-to-end pipeline (`01_ingest.py` … `10_figures.py`) |
| `src/revision/` | Reviewer-response analyses (cited in revised manuscript) |
| `src/` | Analysis pipeline and revision scripts |
| `tests/` | Unit and integration tests |
| `data/published_dataset/` | Dataset README and datasheet (CSVs on Zenodo) |
| `REPRODUCE.md` | Step-by-step reproduction guide |

Manuscript sources and PDFs are kept outside this repository.
| `data/published_dataset/` | Dataset documentation (CSVs on Zenodo at acceptance) |
| `tests/` | Unit tests (`pytest`, ≥85% coverage target) |

Raw inputs (~12 GB) are **not** tracked in git; see `data/raw/README.md` for download instructions.

## Citation

```bibtex
@article{krishnan2026agmigration,
  author  = {Krishnan, Keshav},
  title   = {The Economic Cost of Agricultural Climate Migration in the United States},
