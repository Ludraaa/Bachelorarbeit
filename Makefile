# ============================================================
# Help
# ============================================================

.PHONY: help \
        help-download-cwq \
        help-download-webqsp \
        help-download-wwq \
        help-download-qald7 \
        help-download-qald10 \
        help-download-lcquad2 \
        help-pipeline \
        help-convert \
        help-labels \
        help-prepare \
        help-train \
        help-generate \
        help-resolve \
        help-eval \
        help-demo-qald7-full \
        help-demo-qald7-no-train \
        help-demo-qald7-no-train-no-generate

help:
	@echo "============================================================"
	@echo "Bachelor Thesis Pipeline"
	@echo "============================================================"
	@echo
	@echo "Usage:"
	@echo "  make <target> [RUN_CONFIG=<config>]"
	@echo
	@echo "Setup / datasets:"
	@echo "  download-cwq          Download ComplexWebQuestions"
	@echo "  download-webqsp       Download WebQSP"
	@echo "  download-wwq          Download WebQuestions (Wikidata)"
	@echo "  download-qald7        Download QALD-7"
	@echo "  download-qald10       Download QALD-10"
	@echo "  download-lcquad2      Download LC-QuAD 2"
	@echo
	@echo "Pipeline:"
	@echo "  pipeline              Run the complete pipeline"
	@echo "  convert               Step 1: Convert SPARQL to S-expression"
	@echo "  labels                Step 2: Insert entity/predicate labels"
	@echo "  prepare               Step 3: Prepare data for LLM training"
	@echo "  train                 Step 4: Fine-tune the LLM"
	@echo "  generate              Step 5: Generate predictions"
	@echo "  resolve               Step 6: Resolve entities/predicates and execute queries"
	@echo "  eval                  Step 7: Evaluate prediction quality"
	@echo
	@echo "Demos:"
	@echo "  demo_qald7_full                   Run complete QALD-7 demo"
	@echo "  demo_qald7_no_train               Run QALD-7 demo without training."
	@echo "  demo_qald7_no_train_no_generate   Run QALD-7 demo without training/generation"
	@echo
	@echo "  Note: The full demo requires Qwen2.5-7b to be downloaded to the base model folder. See SETUP.md for more details."
	@echo "  Similarly, the no_train and no_train_no_generate variants can only be run from an uni-freiburg computer, as they require existing data."
	@echo
	@echo "For detailed information about a target:"
	@echo "  make help-<target>"
	@echo
	@echo "Examples:"
	@echo "  make demo_qald7_full"
	@echo "  make pipeline RUN_CONFIG=configs/runs/Wikidata/Qald7/grisp.yaml"
	@echo

# ============================================================
# Detailed help
# ============================================================

help-download-cwq:
	@echo "============================================================"
	@echo "download-cwq"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Download the ComplexWebQuestions dataset from the ChatKBQA repository."
	@echo
	@echo "Reads:"
	@echo "  None."
	@echo
	@echo "Produces:"
	@echo "  \$$(DATA_DIR)/CWQ/origin/CWQ_train.json"
	@echo "  \$$(DATA_DIR)/CWQ/origin/CWQ_dev.json"
	@echo "  \$$(DATA_DIR)/CWQ/origin/CWQ_test.json"
	@echo
	@echo "Runtime:"
	@echo "  A few seconds."
	@echo
	@echo "RAM:"
	@echo "  Negligible."
	@echo
	@echo "Disk:"
	@echo "  Approximately 50MB."
	@echo

help-download-webqsp:
	@echo "============================================================"
	@echo "download-webqsp"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Download the WebQSP dataset from the ChatKBQA repository."
	@echo
	@echo "Reads:"
	@echo "  None."
	@echo
	@echo "Produces:"
	@echo "  \$$(DATA_DIR)/WebQSP/origin/WebQSP_train.json"
	@echo "  \$$(DATA_DIR)/WebQSP/origin/WebQSP_test.json"
	@echo
	@echo "Runtime:"
	@echo "  A few seconds."
	@echo
	@echo "RAM:"
	@echo "  Negligible."
	@echo
	@echo "Disk:"
	@echo "  Approximately 15MB."
	@echo

help-download-wwq:
	@echo "============================================================"
	@echo "download-wwq"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Download the WWQ dataset from https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/wwq/."
	@echo
	@echo "Reads:"
	@echo "  None."
	@echo
	@echo "Produces:"
	@echo "  \$$(DATA_DIR)/WWQ/origin/WWQ_train.jsonl"
	@echo "  \$$(DATA_DIR)/WWQ/origin/WWQ_dev.jsonl"
	@echo "  \$$(DATA_DIR)/WWQ/origin/WWQ_test.jsonl"
	@echo
	@echo "Runtime:"
	@echo "  A few seconds."
	@echo
	@echo "RAM:"
	@echo "  Negligible"
	@echo
	@echo "Disk:"
	@echo "  Approximately 5MB."
	@echo

help-download-qald7:
	@echo "============================================================"
	@echo "download-qald7"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Download the QALD-7 dataset from https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/qald7/."
	@echo
	@echo "Reads:"
	@echo "  None."
	@echo
	@echo "Produces:"
	@echo "  \$$(DATA_DIR)/Qald7/origin/Qald7_train.jsonl"
	@echo "  \$$(DATA_DIR)/Qald7/origin/Qald7_test.jsonl"
	@echo
	@echo "Runtime:"
	@echo "  A few seconds."
	@echo
	@echo "RAM:"
	@echo "  Negligible"
	@echo
	@echo "Disk:"
	@echo "  Approximately 50KB."
	@echo

help-download-qald10:
	@echo "============================================================"
	@echo "download-qald10"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Download the QALD-10 dataset from https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/qald10/."
	@echo
	@echo "Reads:"
	@echo "  None."
	@echo
	@echo "Produces:"
	@echo "  \$$(DATA_DIR)/Qald10/origin/Qald10_train.jsonl"
	@echo "  \$$(DATA_DIR)/Qald10/origin/Qald10_test.jsonl"
	@echo
	@echo "Runtime:"
	@echo "  A few seconds."
	@echo
	@echo "RAM:"
	@echo "  Negligible."
	@echo
	@echo "Disk:"
	@echo "  Approximately 250KB."
	@echo

help-download-lcquad2:
	@echo "============================================================"
	@echo "download-lcquad2"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Download the LC-QuAD 2 dataset from https://ad-publications.cs.uni-freiburg.de/grisp/benchmark/wikidata/lcquad2-new/."
	@echo
	@echo "Reads:"
	@echo "  None."
	@echo
	@echo "Produces:"
	@echo "  \$$(DATA_DIR)/Lcquad2/origin/Lcquad2_train.jsonl"
	@echo "  \$$(DATA_DIR)/Lcquad2/origin/Lcquad2_test.jsonl"
	@echo
	@echo "Runtime:"
	@echo "  A few seconds."
	@echo
	@echo "RAM:"
	@echo "  Negligible"
	@echo
	@echo "Disk:"
	@echo "  Approximately 10MB."
	@echo

help-pipeline:
	@echo "============================================================"
	@echo "pipeline"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Run the complete SPARQL-to-evaluation pipeline."
	@echo
	@echo "Reads:"
	@echo "  RUN_CONFIG=<path to run configuration>"
	@echo "  Dataset files specified by the configuration"
	@echo "  Models/checkpoints specified by the configuration"
	@echo
	@echo "Produces:"
	@echo "  Converted S-expressions"
	@echo "  Label-enriched data"
	@echo "  LLM training data"
	@echo "  Model checkpoints"
	@echo "  Predictions"
	@echo "  Resolved predictions"
	@echo "  Evaluation results"
	@echo
	@echo "Runtime:"
	@echo "  Dataset and config dependent. Anything from minutes to possibly days."
	@echo "  Check the specific pipeline steps for more specific estimates."
	@echo
	@echo "RAM:"
	@echo "  RAM usage is dominated by finetuning, which can be controlled in the configs. GPU usage is strongly recommended."
	@echo
	@echo "Disk:"
	@echo "  A few GB for the fine-tuned adapter, a few hundred MBs for the actual pipeline steps outputs."
	@echo "  Very dependent on dataset and config: If the Resolve Step is running in debug mode, a potentially very big (30GB+) debug file is produced."
	@echo

help-run:
	@echo "============================================================"
	@echo "run"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Alias for 'pipeline'."
	@echo
	@echo "Reads:"
	@echo "  Same as pipeline."
	@echo
	@echo "Produces:"
	@echo "  Same as pipeline."
	@echo
	@echo "Runtime:"
	@echo "  Same as pipeline."
	@echo
	@echo "RAM:"
	@echo "  Same as pipeline."
	@echo
	@echo "Disk:"
	@echo "  Same as pipeline."
	@echo

help-check-run-config:
	@echo "============================================================"
	@echo "check-run-config"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Verify that RUN_CONFIG has been provided. This is just a helper target."
	@echo

help-convert:
	@echo "============================================================"
	@echo "convert"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Convert SPARQL queries to the S-expression representation."
	@echo
	@echo "Reads:"
	@echo "  Dataset files specified by RUN_CONFIG."
	@echo
	@echo "Produces:"
	@echo "  S-expression representations."
	@echo
	@echo "Runtime:"
	@echo "  Dataset size dependent; from a few minutes for the smaller datasets (like WebQSP) to around 30 minutes for something like LC-QuAD2"
	@echo
	@echo "RAM:"
	@echo "  Memory usage generally scales with dataset size; 32GB of total system RAM should be sufficient for the provided datasets."
	@echo
	@echo "Disk:"
	@echo "  Dataset dependent, but typically a few hundred MBs."
	@echo
help-labels:
	@echo "============================================================"
	@echo "labels"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Insert entity and predicate labels into the converted data."
	@echo
	@echo "Reads:"
	@echo "  Converted S-expression data."
	@echo "  Knowledge-base label information."
	@echo
	@echo "Produces:"
	@echo "  Label-enriched dataset files."
	@echo
	@echo "Runtime:"
	@echo "  A few seconds to minutes."
	@echo
	@echo "RAM:"
	@echo "  Memory usage generally scales with dataset size; 32GB of total system RAM should be sufficient for the provided datasets."
	@echo
	@echo "Disk:"
	@echo "  Dataset dependent, but typically a few hundred MBs."
	@echo

help-prepare:
	@echo "============================================================"
	@echo "prepare"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Prepare the processed dataset for LLM fine-tuning."
	@echo
	@echo "Reads:"
	@echo "  Label-enriched dataset."
	@echo
	@echo "Produces:"
	@echo "  LLaMA-Factory training dataset."
	@echo
	@echo "Runtime:"
	@echo "  A few seconds."
	@echo
	@echo "RAM:"
	@echo "  Memory usage generally scales with dataset size; 32GB of total system RAM should be sufficient for the provided datasets."
	@echo
	@echo "Disk:"
	@echo "  A few MBs."
	@echo

help-train:
	@echo "============================================================"
	@echo "train"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Fine-tune the configured language model using LLaMA-Factory."
	@echo
	@echo "Reads:"
	@echo "  LLaMA-Factory training dataset."
	@echo "  Base model."
	@echo "  Training configuration."
	@echo
	@echo "Produces:"
	@echo "  LoRA adapter/model checkpoints."
	@echo
	@echo "Runtime:"
	@echo "  Completely config and hardware dependent; Typically 5-10 hours for medium sized datasets (CWQ, WebQSP)."
	@echo
	@echo "RAM / VRAM:"
	@echo "  Dependant on the specific configuration used. If memory is tight and the model is not gigantic, reducing batch size in the training config can help. All experiments fit within the VRAM of a single NVIDIA L40S GPU."
	@echo
	@echo "Disk:"
	@echo "  Approximately a few GBs for the produced adapter, but also config dependent."
	@echo

help-generate:
	@echo "============================================================"
	@echo "generate"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Generate model prediction beams for the evaluation dataset."
	@echo
	@echo "Reads:"
	@echo "  Fine-tuned model/checkpoint."
	@echo "  Evaluation dataset."
	@echo
	@echo "Produces:"
	@echo "  Raw model prediction beams."
	@echo
	@echo "Runtime:"
	@echo "  Dataset, config and hardware dependent. Could take up to a few days, but typically around the 5-10 hour mark for medium datasets (WebQSP, CWQ)."
	@echo
	@echo "RAM:"
	@echo "  Dependant on the model used. All experiments fit within the VRAM of a single NVIDIA L40S GPU."
	@echo
	@echo "Disk:"
	@echo "  Typically a few hundred MBs."
	@echo

help-resolve:
	@echo "============================================================"
	@echo "resolve"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Resolve predicted entities and predicates and execute the resulting queries."
	@echo
	@echo "Reads:"
	@echo "  Raw model predictions."
	@echo "  Knowledge-base/entity-linking data."
	@echo
	@echo "Produces:"
	@echo "  Resolved predictions."
	@echo
	@echo "Runtime:"
	@echo "  This part of the pipeline has the biggest variance. Heavily influenced by config parameters and especially dataset difficulty (much more significant than raw dataset size.)"
	@echo "  Concrete values for my experimental setup are mentioned in the paper."
	@echo
	@echo "RAM:"
	@echo "  Approximately a few GBs for the base loop, though specific linkers may increase this. The Freebase FACC1 linker, for example, keeps the ~10GB FACC1 index loaded in memory the whole time."
	@echo
	@echo "Disk:"
	@echo "  A few MBs for the actual output, but the optional debug file can exceed 30GBs."
	@echo

help-eval:
	@echo "============================================================"
	@echo "eval"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Evaluate resolved predictions against the gold answers."
	@echo
	@echo "Reads:"
	@echo "  Resolved predictions."
	@echo "  Gold answers."
	@echo
	@echo "Produces:"
	@echo "  Evaluation results."
	@echo
	@echo "Runtime:"
	@echo "  A few minutes."
	@echo
	@echo "RAM:"
	@echo "  Memory usage generally scales with dataset size; 32GB of total system RAM should be sufficient for the provided datasets."
	@echo
	@echo "Disk:"
	@echo "  A few MBs."
	@echo

help-demo-qald7-full:
	@echo "============================================================"
	@echo "demo-qald7-full"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Run the complete QALD-7 Wikidata demonstration pipeline."
	@echo
	@echo "Reads:"
	@echo "  QALD-7 dataset."
	@echo "  QALD-7 run configuration."
	@echo "  Base model."
	@echo
	@echo "Produces:"
	@echo "  Processed datasets."
	@echo "  Fine-tuned model."
	@echo "  Predictions."
	@echo "  Resolved Predictions."
	@echo "  Evaluation results."
	@echo
	@echo "Runtime:"
	@echo "  Should not take longer than 8 hours when using GPU for finetuning and generation."
	@echo
	@echo "RAM / VRAM:"
	@echo "  Experiment fit within the VRAM of a single NVIDIA L40S GPU, as well as 32GB system RAM."
	@echo
	@echo "Disk:"
	@echo "  Approximately 4GB between the adapter and script outputs."
	@echo

help-demo-qald7-no-train:
	@echo "============================================================"
	@echo "demo-qald7-no-train"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Run the QALD-7 demo using an existing trained model."
	@echo
	@echo "Reads:"
	@echo "  QALD-7 dataset."
	@echo "  QALD-7 run configuration."
	@echo "  Existing model/checkpoint."
	@echo
	@echo "Produces:"
	@echo "  Predictions."
	@echo "  Resolved predictions."
	@echo "  Evaluation results."
	@echo
	@echo "Runtime:"
	@echo "  Should not take longer than 8 hours when using GPU for finetuning and generation."
	@echo
	@echo "RAM / VRAM:"
	@echo "  Experiment fit within the VRAM of a single NVIDIA L40S GPU, as well as 32GB system RAM."
	@echo
	@echo "Disk:"
	@echo "  Approximately 200MB of script outputs."
	@echo

help-demo-qald7-no-train-no-generate:
	@echo "============================================================"
	@echo "demo-qald7-no-train-no-generate"
	@echo "============================================================"
	@echo "Description:"
	@echo "  Run QALD-7 resolution and evaluation using existing predictions."
	@echo
	@echo "Reads:"
	@echo "  Existing predictions."
	@echo "  QALD-7 run configuration."
	@echo
	@echo "Produces:"
	@echo "  Resolved predictions."
	@echo "  Evaluation results."
	@echo
	@echo "Runtime:"
	@echo "  Should not take longer than 2 hours."
	@echo
	@echo "RAM:"
	@echo "  A few GB at most."
	@echo
	@echo "Disk:"
	@echo "  Approximately 200MB of script outputs."
	@echo

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

.PHONY: demo-qald7-full demo-qald7-no-train demo-qald7-no-train-no-generate

QALD7_DEMO_CONFIG := configs/runs/Wikidata/Qald7/sparql.yaml
QALD7_DEMO_TRAINING_CONFIG := $(shell $(PYTHON) -c "import yaml; print(yaml.safe_load(open('$(QALD7_DEMO_CONFIG)'))['training_config'])")

demo-qald7-full: download-qald7
	$(PYTHON) src/sparql_to_sexpr.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/insert_labels.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/prepare_llm_data.py --run_config $(QALD7_DEMO_CONFIG)
	lmf train $(QALD7_DEMO_TRAINING_CONFIG)
	$(PYTHON) src/generate_predictions.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/resolve_predictions.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/eval_predictions.py --run_config $(QALD7_DEMO_CONFIG)

demo-qald7-no-train: download-qald7
	$(PYTHON) src/sparql_to_sexpr.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/insert_labels.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/generate_predictions.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/resolve_predictions.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/eval_predictions.py --run_config $(QALD7_DEMO_CONFIG)

demo-qald7-no-train-no-generate:
	$(PYTHON) src/resolve_predictions.py --run_config $(QALD7_DEMO_CONFIG)
	$(PYTHON) src/eval_predictions.py --run_config $(QALD7_DEMO_CONFIG)