```make
# ============================================================
# Configuration
# ============================================================

PYTHON ?= python3

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
        download-lcquad2

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
		-o "$(DATA_DIR)/Qald7/origin/Qald7_test.jsonl"

download-qald10:
	mkdir -p "$(DATA_DIR)/Qald10/origin"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/qald10/train.jsonl" \
		-o "$(DATA_DIR)/Qald10/origin/Qald10_train.jsonl"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/qald10/test.jsonl" \
		-o "$(DATA_DIR)/Qald10/origin/Qald10_test.jsonl"

download-lcquad2:
	mkdir -p "$(DATA_DIR)/Lcquad2/origin"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/lcquad2-new/train.jsonl" \
		-o "$(DATA_DIR)/Lcquad2/origin/Lcquad2_train.jsonl"
	curl -fL "https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/lcquad2-new/test.jsonl" \
		-o "$(DATA_DIR)/Lcquad2/origin/Lcquad2_test.jsonl"


# ============================================================
# Freebase KG Setup
# ============================================================

.PHONY: freebase-install freebase-start freebase-stop

FREEBASE_DIR     ?= Freebase-Setup
VIRTUOSO_PORT    ?= 3001
VIRTUOSO_DB_URL  ?= https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip?dl=1

# Virtuoso Open Source Edition
VIRTUOSO_VERSION ?= 7.2.12
VIRTUOSO_URL     ?= https://github.com/openlink/virtuoso-opensource/archive/refs/tags/v$(VIRTUOSO_VERSION).tar.gz
VIRTUOSO_TARBALL := $(FREEBASE_DIR)/virtuoso-$(VIRTUOSO_VERSION).tar.gz
VIRTUOSO_DIR     := $(FREEBASE_DIR)/virtuoso-opensource

freebase-install:
	@if [ -d "$(FREEBASE_DIR)/.git" ]; then \
		echo "$(FREEBASE_DIR) already cloned, skipping"; \
	else \
		git clone https://github.com/dki-lab/Freebase-Setup.git "$(FREEBASE_DIR)"; \
	fi

	@if [ -d "$(FREEBASE_DIR)/virtuoso_db" ]; then \
		echo "virtuoso_db already extracted, skipping download"; \
	else \
		curl -fL "$(VIRTUOSO_DB_URL)" -o "$(FREEBASE_DIR)/virtuoso_db.zip"; \
		unzip -q "$(FREEBASE_DIR)/virtuoso_db.zip" -d "$(FREEBASE_DIR)"; \
		rm "$(FREEBASE_DIR)/virtuoso_db.zip"; \
	fi

	@if [ -d "$(VIRTUOSO_DIR)" ]; then \
		echo "Virtuoso $(VIRTUOSO_VERSION) already downloaded, skipping"; \
	else \
		curl -fL "$(VIRTUOSO_URL)" -o "$(VIRTUOSO_TARBALL)"; \
		tar -xzf "$(VIRTUOSO_TARBALL)" -C "$(FREEBASE_DIR)"; \
		mv "$(FREEBASE_DIR)/virtuoso-opensource-$(VIRTUOSO_VERSION)" "$(VIRTUOSO_DIR)"; \
		rm "$(VIRTUOSO_TARBALL)"; \
	fi

	@if [ ! -x "$(VIRTUOSO_DIR)/binsrc/virtuoso/virtuoso-t" ]; then \
		echo "Building Virtuoso $(VIRTUOSO_VERSION)..."; \
		cd "$(VIRTUOSO_DIR)" && ./autogen.sh && ./configure && make -j$$(nproc); \
	fi

	@sed -i 's|^virtuosoPath = .*|virtuosoPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "virtuoso-opensource")|' "$(FREEBASE_DIR)/virtuoso.py"

freebase-start:
	cd "$(FREEBASE_DIR)" && $(PYTHON) virtuoso.py start $(VIRTUOSO_PORT) -d virtuoso_db

freebase-stop:
	cd "$(FREEBASE_DIR)" && $(PYTHON) virtuoso.py stop $(VIRTUOSO_PORT)


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


# ============================================================
# Demos (Wikidata / Qald7)
# ============================================================

.PHONY: demo_qald7_full demo_qald7_no_train demo_qald7_no_train_no_generate

QALD7_DEMO_CONFIG := configs/runs/Wikidata/Qald7/grisp.yaml
QALD7_DEMO_TRAINING_CONFIG := $(shell $(PYTHON) -c "import yaml; print(yaml.safe_load(open('$(QALD7_DEMO_CONFIG)'))['training_config'])")

demo_qald7_full: download-qald7
	$(PYTHON) src/sparql_to_sexpr.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/insert_labels.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/prepare_llm_data.py --run_config $(QALD7_DEMO_CONFIG)
	lmf train $(QALD7_DEMO_TRAINING_CONFIG)
	$(PYTHON) src/generate_predictions.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/resolve_predictions.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/eval_predictions.py --run_config $(QALD7_DEMO_CONFIG)

demo_qald7_no_train: download-qald7
	$(PYTHON) src/sparql_to_sexpr.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/insert_labels.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/generate_predictions.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/resolve_predictions.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/eval_predictions.py --run_config $(QALD7_DEMO_CONFIG)

demo_qald7_no_train_no_generate:
	$(PYTHON) src/resolve_predictions.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/eval_predictions.py --run_config $(QALD7_DEMO_CONFIG)
```
