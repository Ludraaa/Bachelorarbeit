# Run Configurations

Run configurations define the parameters of a full pipeline run. They act as a central place to specify the dataset, knowledge base, model configuration, linker configuration, and options for the individual pipeline steps.

Although running the python scripts separately is technically possible, using the Docker + Makefile + run config setup is the intended approach.

A specific run configuration could look something like this:

```yaml
dataset: Qald7
split: test
mode: sparql
kb: wikidata
endpoint_url: "http://amur:7001/sparql"
note: "official run"

entity_linkers: "ChatKBQA.type_map,Wikidata.wbsearchentities_sim"
predicate_linkers: "Wikidata.label_search,Wikidata.neighborhood"

dataset_config: configs/datasets/default.yaml
training_config: configs/training/Wikidata/Qald7/sparql.yaml
infer_config: configs/infer/default.yaml

generate:
  num_beams: 8
  diversity_penalty: 0.4

resolve:
  k1_per_pass: "100,5"
  k2_per_pass: "5,15"
  beam_limits: "15"
  linker_params: {}
  label_fallback: false
  debug: true

eval:
  get_live_gold: true
```

---

## Shared Run Options

The following options are shared between multiple pipeline steps.

### `dataset`

Specifies the dataset to process.

**Type:** `string`

**Required:** Yes

The value is used to locate the corresponding dataset directory under `DATA_DIR`.

For example:

```yaml
dataset: Qald7
```

---

### `split`

Specifies the dataset split to process.

**Type:** `string`

Accepted values are:

* `train`
* `dev`
* `test`

For example:

```yaml
split: test
```

The available splits depend on the dataset.

---

### `mode`

Specifies the target representation used during training format conversion.

**Type:** `string`

**Default:** `sparql`

**Possible values:**

* `jena`: produce the Jena Syntax Expression representation.
* `sparql`: produce normalized SPARQL expression.
* `chatkbqa_cwq`: legacy format of original ChatKBQA for ComplexWebQuestions. Only supported for resolve and eval.
* `chatkbqa_webqsp`: legacy format of original ChatKBQA for WebQuestionSP. Only supported for resolve and eval.


For example:

```yaml
mode: sparql
```

---

### `kb`

Specifies the knowledge-base module used by the pipeline.

**Type:** `string`

**Required:** `Yes`

For example:

```yaml
kb: wikidata
```
A corresponding KB module implementation is dynamically searched for in `src/kb/[kb].py`. Note that this file has to define a subclass of `BaseKB` named `[kb]`, with the first character capitalized.

The KB module provides an abstraction layer for knowledge-base-specific implementations such as KB-specific prefixes or label queries.

---

### `endpoint_url`

Specifies the SPARQL endpoint used for query execution.

**Type:** `string`

**Required:** `Yes`

For example:

```yaml
endpoint_url: "http://amur:7001/sparql"
```

The endpoint is used for operations such as gold-query execution and prediction evaluation.

---

### `note`

Optional free-form description of the run.

**Type:** `string`

For example:

```yaml
note: "official run"
```

A freeform note that can be used to further document certain runs.

---

### `entity_linkers`

Specifies the entity linker combination used during entity retrieval.

**Type:** `string`

**Required:** `Yes`

Multiple linkers are specified as a comma-separated list and are applied in the given order.

For example:

```yaml
entity_linkers: "ChatKBQA.type_map,Wikidata.wbsearchentities_sim"
```

The available linkers are defined by the corresponding KB/linker modules located under `src/linkers/entity/`.

---

### `predicate_linkers`

Specifies the predicate linker combination used during prediction resolution.

**Type:** `string`

**Required:** `Yes`

Multiple linkers are specified as a comma-separated list and are applied in the given order.

For example:

```yaml
predicate_linkers: "Wikidata.label_search,Wikidata.neighborhood"
```

The available linkers are defined by the corresponding KB/linker modules located under `src/linkers/predicate/`.

---

# Configs

The run config refers to a few other configs to keep things organized. These include:

### `dataset_config`

Specifies the path to a dataset configuration.

**Type:** `string`

**Default:** `configs/datasets/default.yaml`

For example:

```yaml
dataset_config: configs/datasets/default.yaml
```
This is used by the conversion step to normalize any dataset into what the further pipeline steps expect.
See the [Dataset config schema](../datasets/schema.md) for the available dataset configuration options.

---

### `training_config`

Specifies the path to a Llama-Factory training configuration used for finetuning.

**Type:** `string`

**Required:** `Yes`

For example:

```yaml
training_config: configs/training/Wikidata/Qald7/sparql.yaml
```

---

### `infer_config`

Specifies the path to a Llama-Factory inference configuration used for inference.

**Type:** `string`

**Default:** `configs/infer/default.yaml`

For example:

```yaml
infer_config: configs/infer/default.yaml
```
The configuration is passed to Llama-Factory when initializing the inference model. It is primarily used to specify model loading and other Llama-Factory-supported inference settings.

Note: The inference procedure itself is largely fixed by the implementation. In particular, generation parameters such as the number of beams and diversity penalty are configured through the generate section of the run configuration and passed explicitly to the Hugging Face generation API. Consequently, Llama-Factory configuration options that affect generation may be overridden by this implementation.

---

# Convert Options

In the conversion step, all items of a dataset are augmented by parsing the gold SPARQL queries into the desired training format. Additionally, this step normalizes the structure of any dataset using a dataset configuration.

The corresponding command-line script is `sparql_to_sexpr.py`.

Most importantly, this step takes a dataset config. More detailed information can be found in the relevant section.

The options below control dataset loading, SPARQL conversion, and gold-query execution.

## `convert`

Conversion-specific options can be placed in the `convert` section of a run configuration.

---

### `no_mismatch_analysis`

Disables comparison between the raw gold SPARQL query and its normalized form.

**Type:** `boolean`

**Default:** `false`

For example:

```yaml
convert:
  no_mismatch_analysis: true
```

By default, the conversion step executes both the original gold query and the normalized gold query and compares their results in order to catch potential normalization errors.
This option is useful for datasets whose raw SPARQL queries cannot be executed directly against the selected endpoint, for example when required prefixes are missing.
When enabled, the raw gold query is not executed. Empty-result detection instead uses the normalized gold query.

---

### `no_gold_exec`

Disables gold-query execution completely.

**Type:** `boolean`

**Default:** `false`

For example:

```yaml
convert:
  no_gold_exec: true
```

When enabled, neither the raw nor normalized gold query is executed against the SPARQL endpoint.
Consequently, no gold answer fields are generated and raw-vs-normalized mismatch analysis is skipped.
This option is intended for cases where any gold-query execution is unnecessary, such as training-only dataset preparation.

**Warning:** Gold execution provides the answers required by later retrieval and evaluation steps. Disabling it makes the resulting augmented dataset unusable for retrieval and evaluation.

`no_gold_exec` effectively implies `no_mismatch_analysis`.

---

# Labels Options

The label insertion step extracts all IRIs and fetches their corresponding natural language labels and type memberships from the knowledge base. This step resolves the labels, caches the results to optimize subsequent runs, and generates the label-enriched dataset files alongside gold entity, relation, and type maps.

The corresponding command-line script for this step is `insert_labels.py`.

The options below control the behavior of the label resolution process.

## `labels`

Label-specific options can be placed in the `labels` section of a run configuration.

---

### `debug`

Enables verbose debug logging and comprehensive failure reporting during label and type resolution.

**Type:** `boolean`

**Default:** `false`

For example:

```yaml
labels:
  debug: true
```

When enabled, the pipeline prints detailed execution traces to the console, including:
* Cache hit, miss, and null statistics.
* SPARQL endpoint query batch progress and sizes.
* Formatter substitutions for individual IRIs.
* A detailed failure report at the end of the run that groups unresolved entity and relation labels by failure reason (e.g., `cached_null`, `sparql_no_label`, or `sparql_error`), listing specific problematic URIs and the IDs of the examples they affect.

---

# Prepare Options

In the dataset preparation step, the pipeline takes the label-enriched data and formats it into the Alpaca-style JSON structure expected by Llama-Factory. It also automatically registers the dataset in Llama-Factory's `dataset_info.json` so it can be referenced directly by name in training configurations.

The corresponding command-line script is `prepare_llm_data.py`.

---

Currently, there are no step-specific configuration options for the `prepare` section. This step relies entirely on the [Shared Run Options](#shared-run-options).

---

# Generation Options

In the generation step, the pipeline uses a trained Llama-Factory model to perform group beam search inference on the target dataset split. It generates candidate Logical Form queries (s-expressions or normalized SPARQL) for each natural language question, deduplicates identical outputs per item while preserving rank order, and records accuracy metrics such as exact matches at rank 0 and overall beam hits against the gold query.

The corresponding command-line script is `generate_predictions.py`.

The options below control beam search, token limits, sample capping, and execution modes during model inference.

## `generate`

Generation-specific options can be placed in the `generate` section of a run configuration.

---

### `num_beams`

Specifies the number of prediction candidate beams to generate per input question using group beam search.

**Type:** `integer`

**Default:** `8`

For example:

```yaml
generate:
  num_beams: 8
```

Higher beam counts can increase the chances of capturing the gold logical form within the predictions, although this has diminishing returns and a runtime cost (both when generating and later resolving).

---

### `diversity_penalty`

Applies a diversity penalty for group beam search across generated candidate sequences.

**Type:** `float`

**Default:** `0.5`

For example:

```yaml
generate:
  diversity_penalty: 0.5
```

A higher diversity penalty forces individual beam groups to explore distinct generation paths, producing a more varied set of candidate queries at the cost of potential coherence drop. In my experiments, recommended values are typically `~1.0` for Llama-based models and `~0.5` for Qwen-based models. Note that these values are not optimal and may vary strongly depending on the actual trained model. They should be treated more as guidelines instead.

---

### `max_new_tokens`

Specifies the maximum number of new tokens the model is allowed to generate for each output query.

**Type:** `integer`

**Default:** `512`

For example:

```yaml
generate:
  max_new_tokens: 512
```

---

### `max_samples`

Caps the total number of dataset entries processed during the generation step.

**Type:** `integer`

**Default:** None (processes the entire dataset split)

For example:

```yaml
generate:
  max_samples: 50
```

This option is primarily used for quick debugging or testing inference configurations. It can be a good idea to initially limit max samples in order to find a good diversity penalty.

---

### `oracle`

Bypasses model inference and directly outputs the gold query as the single prediction.

**Type:** `boolean`

**Default:** `false`

For example:

```yaml
generate:
  oracle: true
```

When set to `true`, model loading and inference are skipped entirely. The script outputs the gold query (`sexpr_with_labels`) as the prediction. This is useful for establishing upper-bound theoretical performance limits in downstream resolution and evaluation steps.

---

# Resolve Options

In the resolution step, the pipeline takes the candidate logical form queries generated during the generation step and maps the predicted entity and predicate placeholders to actual knowledge base identifiers. This is achieved by systematically applying pre-defined entity and predicate linkers across the predicted beams to find an executable SPARQL query.

The corresponding command-line script for this step is `resolve_predictions.py`.

The options below control the behavior of the beam traversal, linker permutation limits, score thresholds, and timeouts during resolution.

## `resolve`

Resolution-specific options can be placed in the `resolve` section of a run configuration.

---

### `k1_per_pass`

Specifies the maximum number of entity permutations generated per beam per predicate linking pass.

**Type:** `string`

**Default:** `"25"`

For example:

```yaml
resolve:
  k1_per_pass: "100,5"

```
In this example, during the first predicate linking pass, entity retrieval will produce a maximum of 100 entity permutations per beam. In the second predicate linking pass, only 5 entity permutations will be produced per beam.
The values are provided as a comma-separated string. If the number of passes (determined by the number of predicate linkers) exceeds the number of provided values, the last value is reused for all subsequent passes.

---

### `t1_per_pass`

Specifies the score threshold for entity permutations per predicate-linker pass. Permutations with a score below this threshold are discarded.

**Type:** `string`

**Default:** `"0.0"`

For example:

```yaml
resolve:
  t1_per_pass: "0.5,0.0"

```

Values are comma-separated. The last value is reused if there are more passes than specified values.

---

### `k2_per_pass`

Specifies the maximum number of predicate permutations generated for every entity permutation per predicate-linker pass.

**Type:** `string`

**Default:** `"5"`

For example:

```yaml
resolve:
  k2_per_pass: "5,15"

```
Assuming the numbers used in this example, during the first predicate linking pass every entity permutation generated by entity retrieval results in a maximum of 5 predicate permutations. For the second predicate
linking pass, this number would instead be 15.
Values are comma-separated. The last value is reused if there are more passes than specified values.

---

### `t2_per_pass`

Specifies the score threshold for predicate permutations per predicate-linker pass.

**Type:** `string`

**Default:** `"0.0"`

For example:

```yaml
resolve:
  t2_per_pass: "0.2,0.0"

```

Values are comma-separated. The last value is reused if there are more passes than specified values.

---

### `beam_limits`

Specifies the maximum number of prediction beams to evaluate per predicate-linker pass.

**Type:** `string`

**Default:** `"8"`

For example:

```yaml
resolve:
  beam_limits: "8,4"

```
This directly limits how many prediction beams a predicate linking pass has access to. This could be used to limit a more expensive linking pass to only the high-confidence beams.
Values are comma-separated. A value of `0` indicates no limit (all available beams will be evaluated). The last value is reused if there are more passes than specified values.

---

### `linker_params`

Provides a JSON string containing custom hyperparameters to override default settings for specific entity or predicate linkers.

**Type:** `string`

**Default:** `"{}"`

For example:

```yaml
resolve:
  linker_params: '{"ChatKBQA.gold_simcse": {"gold_threshold": 0.5}}'

```

The keys in the JSON dictionary must match the linker IDs defined in the shared `entity_linkers` and `predicate_linkers` configuration.

---

### `item_time_limit_sec`

Specifies a total time budget in seconds allocated for resolving a single dataset item.

**Type:** `float`

**Default:** None (no time limit)

For example:

```yaml
resolve:
  item_time_limit_sec: 120.0

```

If resolution of a single item exceeds this limit, processing is aborted and marked as timed out, allowing the pipeline to continue with the next item instead of losing a lot of time on complex queries.

---

### `label_fallback`

Enables a legacy entity label fallback mechanism during SPARQL conversion. This is the label fallback used by the original ChatKBQA and only works on WebQSP and CWQ.

**Type:** `boolean`

**Default:** `false`

For example:

```yaml
resolve:
  label_fallback: true

```

This feature is intended for comparing performance between the extended pipeline and the original.

---

### `debug`

Enables comprehensive debug logging and outputs an additional detailed JSONL trace file.

**Type:** `boolean`

**Default:** `false`

For example:

```yaml
resolve:
  debug: true

```

When enabled, the script outputs step-by-step traces of beam traversal, linking scores, generated permutations, and endpoint execution states to the console. It also saves a `.debug.json` file detailing the specific linker chains, candidates, and queries attempted for every beam and permutation. This debug file can grow very large (40GB+) for certain datasets.

---

# Eval Options

In the evaluation step, the resolved prediction queries are scored against the gold queries. Exact Match, Assignment F1, and Hit@1 are computed by comparing the execution results of the predicted SPARQL queries against the gold answers. Additionally, results are logged to a central ledger containing all run results and performs distribution and hyperparameter sensitivity analysis to evaluate the impact of the hyperparameter configuration.

The corresponding command-line script for this step is `eval_predictions.py`.

The options below control gold answer resolution, execution timeouts, ledger locations, and analysis generation.

## `eval`

Evaluation-specific options can be placed in the `eval` section of a run configuration.

---

### `get_live_gold`

Forces the pipeline to execute the normalized gold SPARQL query against the endpoint to retrieve the gold answers live, rather than relying on answers saved to the dataset.

**Type:** `boolean`

**Default:** `false`

For example:

```yaml
eval:
  get_live_gold: true

```

If the live execution fails or returns empty results, the pipeline falls back to the saved dataset answers unless `live_only` is enabled.
Note that against a stable KB, results saved to the dataset during the convert step should be identical to live query results. If significant time has passed since
converting the dataset or the KB has updated since, using live gold query results is advised.

---

### `live_only`

Ignores saved gold answers entirely. Only items that successfully yield answers from the live gold SPARQL execution are scored.

**Type:** `boolean`

**Default:** `false`

For example:

```yaml
eval:
  get_live_gold: true
  live_only: true

```

Requires `get_live_gold` to be `true`. If enabled, any item that fails to produce live gold answers is treated as "stale" (having no gold answer) and excluded from the final metrics.

---

### `timeout`

Specifies the per-query HTTP timeout in seconds for executing SPARQL queries (both predicted and gold) against the endpoint.

**Type:** `integer`

**Default:** `60`

For example:

```yaml
eval:
  timeout: 120

```

---

### `ledger`

Specifies the file path to the central JSON ledger where the evaluation summary and metrics for this run will be appended.

**Type:** `string`

**Default:** `results/results.json`

For example:

```yaml
eval:
  ledger: "results/experiment_v2.json"

```

---

### `skip_analysis`

Disables the generation of the distribution, performance, and hyperparameter sensitivity analysis reports and plots.

**Type:** `boolean`

**Default:** `false`

For example:

```yaml
eval:
  skip_analysis: true

```

By default, the evaluation step generates a `.analysis.json` file and a folder of plots detailing the F1-impact of reducing specific beam and permutation search caps. Enable this flag to skip this step if you only need the metrics.

---

### `max_samples`

Caps the number of items evaluated during this step.

**Type:** `integer`

**Default:** None (evaluates all resolved items)

For example:

```yaml
eval:
  max_samples: 100

```

Primarily used for debugging the evaluation logic without waiting for the entire dataset to process.