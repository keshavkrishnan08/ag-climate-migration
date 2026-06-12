.PHONY: all env ingest features model switching project stranded cascade insurance frontier figures test clean \
        reproduce headline verify revision-clean rev-help \
        rev-stranded rev-insurance rev-migration rev-yield rev-framework rev-substantive

PYTHON = python
SRC = src
REV_PYTHON = $(PYTHON)
REV_SRC = src/revision
REV_RES = results/revision

all: ingest features model switching project stranded cascade insurance frontier figures test

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

test:
	pytest tests/ --cov=src --cov-fail-under=85

clean:
	rm -rf data/processed/* projections/*

rev-help:
	@echo "  make reproduce    - run all revision experiment scripts"
	@echo "  make headline     - write $(REV_RES)/HEADLINE_NUMBERS.json"
	@echo "  make verify       - headline + print stored vs recomputed"
	@echo "  make revision-clean - remove local revision JSON outputs"

reproduce: rev-stranded rev-insurance rev-migration rev-yield rev-framework rev-substantive

rev-stranded:
	$(REV_PYTHON) $(REV_SRC)/stranded_revision.py
	$(REV_PYTHON) $(REV_SRC)/stranded_floor_sensitivity.py
	$(REV_PYTHON) $(REV_SRC)/hedonic_strengthened.py
	$(REV_PYTHON) $(REV_SRC)/dcf_ci_fixed.py
	$(REV_PYTHON) $(REV_SRC)/dollar_robustness.py

rev-insurance:
	$(REV_PYTHON) $(REV_SRC)/insurance_rolling_aph.py
	$(REV_PYTHON) $(REV_SRC)/insurance_rp_and_tay.py
	$(REV_PYTHON) $(REV_SRC)/insurance_coverage_endogeneity.py
	$(REV_PYTHON) $(REV_SRC)/insurance_sco.py

rev-migration:
	$(REV_PYTHON) $(REV_SRC)/migration_farmdependent.py
	$(REV_PYTHON) $(REV_SRC)/migration_iv_bartik.py
	$(REV_PYTHON) $(REV_SRC)/migration_primeage_panel.py
	$(REV_PYTHON) $(REV_SRC)/migration_wildbootstrap.py
	$(REV_PYTHON) $(REV_SRC)/migration_share_balance.py
	$(REV_PYTHON) $(REV_SRC)/migration_fiscal_chain.py
	$(REV_PYTHON) $(REV_SRC)/migration_depop_montecarlo.py

rev-yield:
	$(REV_PYTHON) $(REV_SRC)/yield_v7_spectrum.py
	$(REV_PYTHON) $(REV_SRC)/yield_audit_target_decomp.py

rev-framework:
	$(REV_PYTHON) $(REV_SRC)/framework_cohesion.py
	$(REV_PYTHON) $(REV_SRC)/framework_common_driver.py

rev-substantive:
	$(REV_PYTHON) $(REV_SRC)/substantive_experiments.py
	$(REV_PYTHON) $(REV_SRC)/tier1_experiments.py
	$(REV_PYTHON) $(REV_SRC)/tier2_experiments.py
	$(REV_PYTHON) $(REV_SRC)/tier3_tighten.py
	$(REV_PYTHON) $(REV_SRC)/tier4_refit.py
	$(REV_PYTHON) $(REV_SRC)/tier5_residuals.py

headline:
	$(REV_PYTHON) $(REV_SRC)/headline_numbers.py

verify: headline
	@$(REV_PYTHON) -c "import json; d=json.load(open('$(REV_RES)/HEADLINE_NUMBERS.json')); [print(f'  {k:<42} stored={v.get(\"value\"):<10}  recomputed={v.get(\"value_recomputed\")}') for k,v in d.items() if isinstance(v,dict) and 'value_recomputed' in v]"

revision-clean:
	rm -f $(REV_RES)/*.json
