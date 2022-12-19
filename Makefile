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
REV_SRC    = src/revision
REV_RES    = results/revision

rev-help:
	@echo "Revision targets:"
	@echo "  make reproduce       - run every revision-headline + robustness script"
	@echo "  make headline        - consolidate cited numbers -> HEADLINE_NUMBERS.json"
	@echo "  make verify          - run headline; show cited vs recomputed"
	@echo "  make revision-paper  - recompile main + SI + response + tracked-changes"
	@echo "  make revision-clean  - remove regenerated revision JSONs and paper artifacts"

reproduce: rev-stranded rev-insurance rev-migration rev-yield rev-framework rev-substantive

rev-stranded:
	@echo "[stranded] DCF + alternate-use floor + hedonic + CI + ML/process"
	$(REV_PYTHON) $(REV_SRC)/stranded_revision.py
	$(REV_PYTHON) $(REV_SRC)/stranded_floor_sensitivity.py
	$(REV_PYTHON) $(REV_SRC)/hedonic_strengthened.py
	$(REV_PYTHON) $(REV_SRC)/dcf_ci_fixed.py
	$(REV_PYTHON) $(REV_SRC)/dollar_robustness.py

rev-insurance:
	@echo "[insurance] rolling-APH + TAY/YE + RP put + coverage + SCO"
	$(REV_PYTHON) $(REV_SRC)/insurance_rolling_aph.py
	$(REV_PYTHON) $(REV_SRC)/insurance_rp_and_tay.py
	$(REV_PYTHON) $(REV_SRC)/insurance_coverage_endogeneity.py
	$(REV_PYTHON) $(REV_SRC)/insurance_sco.py

rev-migration:
	@echo "[migration] shift-share IV + prime-age + wild-cluster + share-balance + depop MC"
	$(REV_PYTHON) $(REV_SRC)/migration_farmdependent.py
	$(REV_PYTHON) $(REV_SRC)/migration_iv_bartik.py
	$(REV_PYTHON) $(REV_SRC)/migration_primeage_panel.py
	$(REV_PYTHON) $(REV_SRC)/migration_wildbootstrap.py
	$(REV_PYTHON) $(REV_SRC)/migration_share_balance.py
	$(REV_PYTHON) $(REV_SRC)/migration_fiscal_chain.py
	$(REV_PYTHON) $(REV_SRC)/migration_depop_montecarlo.py

rev-yield:
	@echo "[yield] spectrum model + target-vs-feature decomposition"
	$(REV_PYTHON) $(REV_SRC)/yield_v7_spectrum.py
	$(REV_PYTHON) $(REV_SRC)/yield_audit_target_decomp.py

rev-framework:
	@echo "[framework] common-cause test + chain-test (old)"
	$(REV_PYTHON) $(REV_SRC)/framework_cohesion.py
	$(REV_PYTHON) $(REV_SRC)/framework_common_driver.py

rev-substantive:
	@echo "[substantive] E1-E9 + tier-1 (E10-E30) + tier-2 (E31-E45) experiment battery"
	$(REV_PYTHON) $(REV_SRC)/substantive_experiments.py
	$(REV_PYTHON) $(REV_SRC)/tier1_experiments.py
	$(REV_PYTHON) $(REV_SRC)/tier2_experiments.py

headline:
	@echo "[headline] consolidating every cited number"
	$(REV_PYTHON) $(REV_SRC)/headline_numbers.py

verify: headline
	@echo ""
	@$(REV_PYTHON) -c "import json; d=json.load(open('$(REV_RES)/HEADLINE_NUMBERS.json')); [print(f'  {k:<42} cited={v.get(\"value\"):<10}  recomputed={v.get(\"value_recomputed\")}') for k,v in d.items() if isinstance(v,dict) and 'value_recomputed' in v]"

revision-paper:
	@echo "[paper] recompiling main + SI + response + tracked-changes"
	cd paper/revision && \
	  for doc in main supplementary_information response_to_reviewers cover_letter_revision; do \
	    for i in 1 2 3; do pdflatex -interaction=nonstopmode $$doc.tex >/dev/null 2>&1; bibtex $$doc >/dev/null 2>&1; done; \
	    echo "  built $$doc.pdf"; \
	  done

revision-clean:
	rm -f results/revision/*.json
	rm -f paper/revision/*.aux paper/revision/*.log paper/revision/*.bbl paper/revision/*.blg paper/revision/*.out paper/revision/*.toc paper/revision/*.fls paper/revision/*.fdb_latexmk
	@echo "Cleaned revision JSONs and LaTeX artifacts. Raw data and parquets preserved."
