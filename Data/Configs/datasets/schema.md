# Dataset Configurations

Dataset configurations describe how raw dataset files are transformed into the format expected by the pipeline.

## Dataset Format

The `format` parameter describes the general structure of the raw dataset.

### `format: flat`

A flat dataset is a list in which each dataset item contains all relevant information at the same level.

For example:

```yaml
- ID: example-1
  question: "Who wrote ...?"
  sparql: "SELECT ..."
- ID: example-2
  question: "Where is ...?"
  sparql: "SELECT ..."
```

For flat datasets, the `fields` option can be used to map dataset-specific field names to the names expected by the pipeline.

### `format: nested`

A nested dataset contains multiple levels of objects. Typically, information that identifies a question is stored in an outer object, while one or more nested objects contain possible parses or other information associated with that question.

For example, WebQSP stores the question and its possible parses separately:

```yaml
Questions:
  - QuestionId: example-1
    RawQuestion: "Who wrote ...?"
    Parses:
      - ParseId: 0
        Sparql: "SELECT ..."
      - ParseId: 1
        Sparql: "SELECT ..."
```

Setting `format` to `nested` allows these structures to be converted into the flat representation expected by the pipeline. The `root`, `nested`, `parse_strategy`, and `inherit` options control how this conversion is performed.

**Possible values:** `flat`, `nested`

**Default:** `flat`

---

## Nested Dataset Options

The following options are only used when `format` is set to `nested`.

### `root`

Specifies the key containing the list of parent objects.

**Type:** `string`

**Required:** Yes

For the WebQSP example above:

```yaml
root: Questions
```

### `nested`

Specifies the key containing the list of child objects within each parent object.

**Type:** `string`

**Required:** Yes

For the WebQSP example:

```yaml
nested: Parses
```

### `parse_strategy`

Determines which nested objects are converted into dataset entries.

**Possible values:**

* `all`: create one dataset entry for every nested object.
* `first`: use only the first nested object of each parent.

**Default:** `all`

For example, with:

```yaml
parse_strategy: first
```

only the first parse of each WebQSP question is retained.

---

## Inherited Fields

The `inherit` option copies fields from the parent object of a nested dataset into the resulting dataset entry.

The mapping uses the following format:

```yaml
inherit:
  SourceField: target_field
```

The source field is read from the parent object, while the target field is written to the resulting flat dataset entry.

**Type:** `mapping`

**Default:** `{}`

For example:

```yaml
inherit:
  QuestionId: question_id
  RawQuestion: question
```

converts the parent fields `QuestionId` and `RawQuestion` into the common fields `question_id` and `question`.

This option is only used with `format: nested`.

---

## Field Mapping

The `fields` option maps fields from the input dataset to the field names expected by the pipeline.

The basic form is:

```yaml
fields:
  SourceField: target_field
```

For example:

```yaml
fields:
  ParseId: id
  Sparql: sparql
```

renames `ParseId` to `id` and `Sparql` to `sparql`.

**Type:** `mapping`

**Default:** `{}`

The same option can also be used to extract a specific field from dictionaries contained in a list. In this form, the mapping specifies both `name` and `extract`:

```yaml
fields:
  Answers:
    name: answer
    extract: AnswerArgument
```

Here, `Answers` is expected to contain a list of dictionaries. The value of `AnswerArgument` is extracted from each dictionary and stored in the resulting `answer` field.
