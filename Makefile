.PHONY: all env ingest features model switching project stranded cascade insurance frontier figures paper test clean

PYTHON = python
SRC = src
RESULTS = results/$(shell date +%Y%m%d_%H%M%S)

all: ingest features model switching project stranded cascade insurance frontier figures paper test

env:
	conda env create -f environment.yml

ingest:
	$(PYTHON) $(SRC)/01_ingest.py

features:
	$(PYTHON) $(SRC)/02_features.py

model:
	$(PYTHON) $(SRC)/03_yield_model.py

switching:
	$(PYTHON) $(SRC)/04_switching.py

project:
	$(PYTHON) $(SRC)/05_project.py

stranded:
	$(PYTHON) $(SRC)/06_stranded.py

cascade:
	$(PYTHON) $(SRC)/07_cascade.py

insurance:
	$(PYTHON) $(SRC)/08_insurance.py

frontier:
	$(PYTHON) $(SRC)/09_frontier.py

figures:
	$(PYTHON) $(SRC)/10_figures.py

paper:
	cd paper && latexmk -pdf main.tex

test:
	pytest tests/ --cov=src --cov-fail-under=85

clean:
	rm -rf data/processed/* projections/*

# ============================================================
# REVISION PIPELINE
# Reproduces every number cited in the revised manuscript.
# See REPRODUCE.md for the full headline-number -> script map.
# ============================================================

.PHONY: reproduce headline verify revision-paper revision-clean rev-help \
        rev-stranded rev-insurance rev-migration rev-yield rev-framework rev-substantive

REV_PYTHON = /opt/anaconda3/bin/python3
