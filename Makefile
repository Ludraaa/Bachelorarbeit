RUN_CONFIG ?=

ifeq ($(RUN_CONFIG),)
$(error RUN_CONFIG is not set. Usage: make pipeline RUN_CONFIG=configs/run/<name>.yaml)
endif

PYTHON ?= python

TRAINING_CONFIG := $(shell $(PYTHON) -c "import yaml,sys; print(yaml.safe_load(open('$(RUN_CONFIG)'))['training_config'])")

.PHONY: pipeline run convert labels prepare train generate resolve eval

pipeline: convert labels prepare train generate resolve eval

run: pipeline

convert:
	$(PYTHON) src/sparql_to_sexpr.py --run_config $(RUN_CONFIG)

labels:
	$(PYTHON) src/insert_labels.py --run_config $(RUN_CONFIG)

prepare:
	$(PYTHON) src/prepare_llm_data.py --run_config $(RUN_CONFIG)

train:
	lmf train $(TRAINING_CONFIG)

generate:
	$(PYTHON) src/generate_predictions.py --run_config $(RUN_CONFIG)

resolve:
	$(PYTHON) src/resolve_predictions.py --run_config $(RUN_CONFIG)

eval:
	$(PYTHON) src/eval_predictions.py --run_config $(RUN_CONFIG)