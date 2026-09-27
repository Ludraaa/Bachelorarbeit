import pytest
import yaml
import argparse

from src.utils.run_config import (
    _default_name,
    _normalize,
    model_id_from_training_config,
    apply_run_config_defaults,
    require,
    validate_choice
)

RUN_CONFIG = """
dataset: WebQSP
split: test
mode: sparql
kb: freebase
endpoint_url: "http://localhost:3001/sparql"
note: "official run"

entity_linkers: "ChatKBQA.type_map,ChatKBQA.facc1"
predicate_linkers: "ChatKBQA.simple,ChatKBQA.neighborhood"

dataset_config: configs/datasets/WebQSP.yaml
training_config: configs/training/Freebase/WebQSP/Llama-2-7b_sparql.yaml
infer_config: configs/infer/default.yaml

generate:
  num_beams: 15
  diversity_penalty: 1.0

resolve:
  k1_per_pass: "500,50"
  k2_per_pass: "1,4000"
  beam_limits: "15"
  linker_params: {}
  label_fallback: false
  debug: true
  item_time_limit_sec: 1200

eval:
  get_live_gold: true
"""

@pytest.mark.parametrize(
    "name, path, expected",
    [
        (
            "Simple",
            "/some/directory/that/is/important/configs/run/test.yaml",
            "test"
        ),
        (
            "Nested",
            "/some/directory/that/is/important/configs/run/Wikidata/tests/test1.yaml",
            "Wikidata_tests_test1"
        ),
    ],
)
def test_default_name(name, path, expected):
    assert _default_name(path) == expected


@pytest.mark.parametrize(
    "name, key, value, expected",
    [
        (
            "Simple",
            "dataset",
            "dataset",
            "dataset"
        ),
        (
            "List",
            "entity_linkers",
            ["SomeLinker1", "SomeLinker2"],
            "SomeLinker1,SomeLinker2"
        ),
        (
            "Dict",
            "linker_params",
            {"SomeLinker1": {"SomeParam1": "SomeString1", "SomeParam2": 51}},
            '{"SomeLinker1": {"SomeParam1": "SomeString1", "SomeParam2": 51}}'
        ),
        (
            "None",
            "doesnotexist",
            None,
            None
        )
    ],
)
def test_normalize(name, key, value, expected):
    assert _normalize(key, value) == expected


@pytest.mark.parametrize(
    "name, config, expected",
    [
        (
            "Output Dir",
            {
                "output_dir": "/some/path/MyModel",
                "model_name_or_path": "/some/path/BaseModel",
            },
            "MyModel",
        ),
        (
            "Model name or path",
            {
                "model_name_or_path": "/some/path/BaseModel",
            },
            "BaseModel",
        ),
    ],
)
def test_model_id_from_training_config(tmp_path, name, config, expected):
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert model_id_from_training_config(str(config_path)) == expected
    
    
def test_apply_run_config_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(RUN_CONFIG, encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--mode", default="sparql")
    parser.add_argument("--kb")
    parser.add_argument("--entity_linkers")
    parser.add_argument("--predicate_linkers")
    parser.add_argument("--k1_per_pass")
    parser.add_argument("--k2_per_pass")
    parser.add_argument("--beam_limits")
    parser.add_argument("--linker_params")
    parser.add_argument("--label_fallback", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--item_time_limit_sec", type=int)

    monkeypatch.setattr(
        "sys.argv",
        ["test", "--run_config", str(config_path)],
    )

    apply_run_config_defaults(parser, section="resolve")

    args = parser.parse_args()

    assert args.dataset == "WebQSP"
    assert args.split == "test"
    assert args.mode == "sparql"
    assert args.kb == "freebase"
    assert args.entity_linkers == "ChatKBQA.type_map,ChatKBQA.facc1"
    assert args.predicate_linkers == "ChatKBQA.simple,ChatKBQA.neighborhood"

    assert args.k1_per_pass == "500,50"
    assert args.k2_per_pass == "1,4000"
    assert args.beam_limits == "15"
    assert args.linker_params == "{}"
    assert args.label_fallback is False
    assert args.debug is True
    assert args.item_time_limit_sec == 1200
    
    require(args, "dataset", "split", "mode")
    validate_choice(args, "mode", ["jena", "sparql"])
    
    
def test_require_rejects_missing_argument():
    args = argparse.Namespace(dataset="WebQSP", split=None, mode="sparql")

    with pytest.raises(SystemExit):
        require(args, "dataset", "split", "mode")
        
        
def test_validate_choice_rejects_invalid_choice():
    args = argparse.Namespace(mode="invalid")

    with pytest.raises(SystemExit):
        validate_choice(args, "mode", ["jena", "sparql"])