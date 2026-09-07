# ============================================================
# Configuration
# ============================================================

PYTHON ?= python

# Hugging Face model to download.
#
# Usage:
#   make hf-model HF_MODEL=meta-llama/Llama-2-7b-hf
#
HF_MODEL ?=

HF_MODEL_DIR = $(LLM_DIR)/Models/$(subst /,--,$(HF_MODEL))


# ============================================================
# Pipeline configuration
# ============================================================

ifneq ($(RUN_CONFIG),)

TRAINING_CONFIG := $(shell $(PYTHON) -c "import yaml; print(yaml.safe_load(open('$(RUN_CONFIG)'))['training_config'])")

ENDPOINT_URL := $(shell $(PYTHON) -c "import yaml; print(yaml.safe_load(open('$(RUN_CONFIG)')).get('endpoint_url', ''))")
export ENDPOINT_URL

endif


# ============================================================
# Setup
# ============================================================

.PHONY: download-cwq \
        download-webqsp \
        download-wwq \
        download-qald7 \
        download-qald10 \
        download-spinach \
        download-lcquad2 \
		download-wdql

download-cwq:
	mkdir -p "$(DATA_DIR)/CWQ/origin"
	curl -fL "https://raw.githubusercontent.com/LHRLAB/ChatKBQA/main/data/CWQ/origin/ComplexWebQuestions_train.json" \
		-o "$(DATA_DIR)/CWQ/origin/CWQ_train.json"
	curl -fL "https://raw.githubusercontent.com/LHRLAB/ChatKBQA/main/data/CWQ/origin/ComplexWebQuestions_dev.json" \
		-o "$(DATA_DIR)/CWQ/origin/CWQ_dev.json"
	curl -fL "https://raw.githubusercontent.com/LHRLAB/ChatKBQA/main/data/CWQ/origin/ComplexWebQuestions_test.json" \
		-o "$(DATA_DIR)/CWQ/origin/CWQ_test.json"

download-webqsp:
	mkdir -p "$(DATA_DIR)/WebQSP/origin"
	curl -fL "https://raw.githubusercontent.com/LHRLAB/ChatKBQA/main/data/WebQSP/origin/WebQSP.train.json" \
		-o "$(DATA_DIR)/WebQSP/origin/WebQSP_train.json"
	curl -fL "https://raw.githubusercontent.com/LHRLAB/ChatKBQA/main/data/WebQSP/origin/WebQSP.test.json" \
		-o "$(DATA_DIR)/WebQSP/origin/WebQSP_test.json"

download-wwq:
	mkdir -p "$(DATA_DIR)/WWQ/origin"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/wwq/train.jsonl" \
		-o "$(DATA_DIR)/WWQ/origin/WWQ_train.jsonl"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/wwq/val.jsonl" \
		-o "$(DATA_DIR)/WWQ/origin/WWQ_dev.jsonl"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/wwq/test.jsonl" \
		-o "$(DATA_DIR)/WWQ/origin/WWQ_test.jsonl"

download-qald7:
	mkdir -p "$(DATA_DIR)/Qald7/origin"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/qald7/train.jsonl" \
		-o "$(DATA_DIR)/Qald7/origin/Qald7_train.jsonl"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/qald7/test.jsonl" \
		-o "$(DATA_DIR)/Qald7/origin/Qald_test.jsonl"

download-qald10:
	mkdir -p "$(DATA_DIR)/Qald10/origin"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/qald10/train.jsonl" \
		-o "$(DATA_DIR)/Qald10/origin/Qald10_train.jsonl"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/qald10/test.jsonl" \
		-o "$(DATA_DIR)/Qald10/origin/Qald10_test.jsonl"

download-spinach:
	mkdir -p "$(DATA_DIR)/Spinach/origin"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/spinach/val.jsonl" \
		-o "$(DATA_DIR)/Spinach/origin/Spinach_dev.jsonl"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/spinach/test.jsonl" \
		-o "$(DATA_DIR)/Spinach/origin/Spinach_test.jsonl"

download-lcquad2:
	mkdir -p "$(DATA_DIR)/Lcquad2/origin"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/lcquad2-new/train.jsonl" \
		-o "$(DATA_DIR)/Lcquad2/origin/Lcquad2_train.jsonl"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/lcquad2-new/test.jsonl" \
		-o "$(DATA_DIR)/Lcquad2/origin/Lcquad2_test.jsonl"

download-wdql:
	mkdir -p "$(DATA_DIR)/WDQL/origin"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/wdql/train.jsonl" \
		-o "$(DATA_DIR)/WDQL/origin/WDQL_train.jsonl"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/wdql/val.jsonl" \
		-o "$(DATA_DIR)/WDQL/origin/WDQL_dev.jsonl"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/wdql/test.jsonl" \
		-o "$(DATA_DIR)/WDQL/origin/WDQL_test.jsonl"

# ============================================================
# Pipeline
# ============================================================

.PHONY: pipeline run convert labels prepare train generate resolve eval

pipeline: check-run-config convert labels prepare train generate resolve eval

run: pipeline

check-run-config:
ifeq ($(RUN_CONFIG),)
	$(error RUN_CONFIG is not set. Usage: make pipeline RUN_CONFIG=configs/run/<name>.yaml)
endif

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