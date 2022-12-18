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
