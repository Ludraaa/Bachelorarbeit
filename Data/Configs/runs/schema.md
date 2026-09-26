# Run Configurations

Run configurations define the parameters of a full pipeline run. They act as a central place to specify the dataset, knowledge base, model configuration, linker configuration, and options for the individual pipeline steps.

Although running the python scripts seperately is technically possible, using the Docker + Makefile + run config setup is the intended approach.

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
See the **Dataset Configurations** schema for the available dataset configuration options.

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

# Conversion Options

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

