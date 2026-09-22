import pytest
import json
from src.utils.kb import load_kb_module

from src.sparql_to_sexpr import (
    get_split_files,
    build_output_path,
    _apply_field_mapping,
    extract_flat_entries,
    extract_nested_entries,
)

def test_get_split_files(tmp_path, monkeypatch):
    origin = tmp_path / "WebQSP" / "origin"
    origin.mkdir(parents=True)
    
    (origin / "WebQSP_test.json").touch()
    (origin / "WebQSP_train.json").touch()
    
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    result = get_split_files("WebQSP")

    assert result == [
        ("test", str(origin / "WebQSP_test.json")),
        ("train", str(origin / "WebQSP_train.json")),
    ]
    
    
def test_build_output_path(tmp_path, monkeypatch):    
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    
    result = build_output_path("WebQSP", "test", "sparql")
    
    assert result == str(tmp_path / "WebQSP" / "sexpr" / "WebQSP_test.sparql.expr.json")
    

def test_apply_field_mappings():
    inp = {"some": "dictionary", "i": "came", "up": "with"}
    mapping = {"some": "one", "i": "you", "up": "up"}
    
    assert _apply_field_mapping(inp, mapping) == {"one": "dictionary", "you": "came", "up": "with",}


@pytest.mark.parametrize(
    "name, data, config, expected",
    [
        (
            "plain list",
            [
                {"id": 1, "question": "What is X?"},
                {"id": 2, "question": "What is Y?"},
            ],
            {
                "format": "flat",
            },
            [
                {"id": 1, "question": "What is X?"},
                {"id": 2, "question": "What is Y?"},
            ],
        ),
        (
            "dict with meta block",
            {
                "meta": "something",
                "items": [
                    {"id": 1},
                    {"id": 2},
                ],
            },
            {
                "format": "flat",
                "root": "items",
            },
            [
                {"id": 1},
                {"id": 2},
            ],
        ),
        (
            "rename fields",
            [
                {"question_id": 1, "text": "What is X?"},
            ],
            {
                "format": "flat",
                "fields": {
                    "question_id": "id",
                    "text": "question",
                },
            },
            [
                {"id": 1, "question": "What is X?"},
            ],
        ),
        (
            "extract field from list of dictionaries",
            [
                {
                    "id": 1,
                    "answers": [
                        {"answer": "Alice"},
                        {"answer": "Bob"},
                    ],
                },
            ],
            {
                "format": "flat",
                "fields": {
                    "id": "id",
                    "answers": {
                        "name": "answers",
                        "extract": "answer",
                    },
                },
            },
            [
                {
                    "id": 1,
                    "answers": ["Alice", "Bob"],
                },
            ],
        ),
        (
            "missing field",
            [
                {"id": 1},
            ],
            {
                "format": "flat",
                "fields": {
                    "question": "question",
                },
            },
            [
                {"question": None},
            ],
        ),
    ],
)
def test_extract_flat_entries(name, data, config, expected):
    assert extract_flat_entries(data, config) == expected
    

@pytest.mark.parametrize(
    "name, data, config, expected",
    [
        (
            "first child with inherited field",
            [
                {
                    "id": 1,
                    "question": "What is X?",
                    "answers": [
                        {"text": "Alice"},
                        {"text": "Bob"},
                    ],
                }
            ],
            {
                "format": "nested",
                "nested": "answers",
                "parse_strategy": "first",
                "inherit": {
                    "id": "question_id",
                },
                "fields": {
                    "text": "answer",
                },
            },
            [
                {
                    "question_id": 1,
                    "answer": "Alice",
                }
            ],
        ),
        (
            "all children with inherited field",
            [
                {
                    "id": 1,
                    "answers": [
                        {"text": "Alice"},
                        {"text": "Bob"},
                    ],
                }
            ],
            {
                "format": "nested",
                "nested": "answers",
                "parse_strategy": "all",
                "inherit": {
                    "id": "question_id",
                },
                "fields": {
                    "text": "answer",
                },
            },
            [
                {
                    "question_id": 1,
                    "answer": "Alice",
                },
                {
                    "question_id": 1,
                    "answer": "Bob",
                },
            ],
        ),
        (
            "nested data with root",
            {
                "questions": [
                    {
                        "id": 1,
                        "answers": [
                            {"text": "Alice"},
                        ],
                    }
                ]
            },
            {
                "format": "nested",
                "root": "questions",
                "nested": "answers",
                "parse_strategy": "all",
                "inherit": {
                    "id": "question_id",
                },
                "fields": {
                    "text": "answer",
                },
            },
            [
                {
                    "question_id": 1,
                    "answer": "Alice",
                }
            ],
        ),
        (
            "field extraction from nested child",
            [
                {
                    "id": 1,
                    "answers": [
                        {
                            "text": "Alice",
                            "metadata": [
                                {"value": "person"},
                                {"value": "human"},
                            ],
                        }
                    ],
                }
            ],
            {
                "format": "nested",
                "nested": "answers",
                "parse_strategy": "first",
                "inherit": {
                    "id": "question_id",
                },
                "fields": {
                    "text": "answer",
                    "metadata": {
                        "name": "types",
                        "extract": "value",
                    },
                },
            },
            [
                {
                    "question_id": 1,
                    "answer": "Alice",
                    "types": ["person", "human"],
                }
            ],
        ),
    ],
)
def test_extract_nested_entries(name, data, config, expected):
    assert extract_nested_entries(data, config) == expected