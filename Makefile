.PHONY: all pipeline env ingest features model switching project stranded cascade insurance frontier figures test clean \
        pipeline-help pipeline-clean \
        stranded-dcf insurance-decomp migration-analysis yield-skill framework-tests robustness summary figures-extra verify

PYTHON = python
SRC = src
STAGE_SRC = src/revision
RESULTS = results/revision

all: pipeline test

env:
	conda env create -f environment.yml

pipeline: ingest features model switching project stranded cascade insurance frontier figures \
          stranded-dcf insurance-decomp migration-analysis yield-skill framework-tests robustness summary

pipeline-help:
	@echo "Pipeline stages (run individually or use make pipeline):"
	@echo "  ingest features model switching project  - data and yield projections"
	@echo "  stranded cascade insurance frontier figures - core economic modules"
	@echo "  stranded-dcf insurance-decomp migration-analysis yield-skill"
	@echo "  framework-tests robustness summary"
	@echo "  make verify  - rebuild HEADLINE_NUMBERS.json and check values"

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

stranded-dcf:
	$(PYTHON) $(STAGE_SRC)/stranded_revision.py
	$(PYTHON) $(STAGE_SRC)/stranded_floor_sensitivity.py
	$(PYTHON) $(STAGE_SRC)/hedonic_strengthened.py
	$(PYTHON) $(STAGE_SRC)/dcf_ci_fixed.py
	$(PYTHON) $(STAGE_SRC)/dollar_robustness.py

insurance-decomp:
	$(PYTHON) $(STAGE_SRC)/insurance_rolling_aph.py
	$(PYTHON) $(STAGE_SRC)/insurance_rp_and_tay.py
	$(PYTHON) $(STAGE_SRC)/insurance_coverage_endogeneity.py
	$(PYTHON) $(STAGE_SRC)/insurance_sco.py

migration-analysis:
	$(PYTHON) $(STAGE_SRC)/migration_farmdependent.py
	$(PYTHON) $(STAGE_SRC)/migration_iv_bartik.py
	$(PYTHON) $(STAGE_SRC)/migration_primeage_panel.py
	$(PYTHON) $(STAGE_SRC)/migration_wildbootstrap.py
	$(PYTHON) $(STAGE_SRC)/migration_share_balance.py
	$(PYTHON) $(STAGE_SRC)/migration_fiscal_chain.py
	$(PYTHON) $(STAGE_SRC)/migration_depop_montecarlo.py

yield-skill:
	$(PYTHON) $(STAGE_SRC)/yield_v7_spectrum.py
	$(PYTHON) $(STAGE_SRC)/yield_audit_target_decomp.py

framework-tests:
	$(PYTHON) $(STAGE_SRC)/framework_cohesion.py
	$(PYTHON) $(STAGE_SRC)/framework_common_driver.py

robustness:
	$(PYTHON) $(STAGE_SRC)/substantive_experiments.py
	$(PYTHON) $(STAGE_SRC)/tier1_experiments.py
	$(PYTHON) $(STAGE_SRC)/tier2_experiments.py
	$(PYTHON) $(STAGE_SRC)/tier3_tighten.py
	$(PYTHON) $(STAGE_SRC)/tier4_refit.py
	$(PYTHON) $(STAGE_SRC)/tier5_residuals.py
	$(PYTHON) $(STAGE_SRC)/robustness_battery.py

figures-extra:
	$(PYTHON) $(STAGE_SRC)/supplementary_figures.py
	$(PYTHON) $(STAGE_SRC)/si_graphics.py

summary:
	$(PYTHON) $(STAGE_SRC)/headline_numbers.py

verify: summary
	@$(PYTHON) -c "import json; d=json.load(open('$(RESULTS)/HEADLINE_NUMBERS.json')); [print(f'  {k:<42} stored={v.get(\"value\"):<10}  recomputed={v.get(\"value_recomputed\")}') for k,v in d.items() if isinstance(v,dict) and 'value_recomputed' in v]"

pipeline-clean:
	rm -f $(RESULTS)/*.json $(RESULTS)/supplementary/*.json
