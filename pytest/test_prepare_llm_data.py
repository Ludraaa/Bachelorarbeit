import json
from types import SimpleNamespace

import pytest

import src.prepare_llm_data as prepare_llm_data


@pytest.fixture
def args():
    return SimpleNamespace(dataset="WebQSP")


def test_load_data(tmp_path, monkeypatch, args):
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / args.dataset / "generation" / "merged"
    dataset_dir.mkdir(parents=True)

    monkeypatch.setenv("DATA_DIR", str(data_dir))

    with open(dataset_dir / "WebQSP_train.jena.json", "w") as f:
        json.dump([{"id": 1}], f)

    with open(dataset_dir / "WebQSP_train.sparql.json", "w") as f:
        json.dump([{"id": 2}], f)

    assert prepare_llm_data.load_data("train", args) == {
        "jena": [{"id": 1}],
        "sparql": [{"id": 2}],
    }


def test_load_data_no_files(tmp_path, monkeypatch, args):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError):
        prepare_llm_data.load_data("train", args)


def test_prepare_dataloader(tmp_path, monkeypatch, args):
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / args.dataset / "generation" / "merged"
    dataset_dir.mkdir(parents=True)

    llm_dir = tmp_path / "llm"

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("LLM_DIR", str(llm_dir))
    monkeypatch.setattr(
        prepare_llm_data,
        "DATASET_INFO_PATH",
        str(llm_dir / "data" / "dataset_info.json"),
    )

    data = [
        {
            "question": "Who was Albert Einstein?",
            "sexpr_with_labels": "(JOIN fb\\:Albert_Einstein ...)",
        },
        {
            "question": "Empty",
            "sexpr_with_labels": "",
        },
    ]

    with open(dataset_dir / "WebQSP_train.jena.json", "w") as f:
        json.dump(data, f)

    prepare_llm_data.prepare_dataloader(args, "train")

    output = llm_dir / "data" / "WebQSP_train.jena" / "examples.json"

    with open(output) as f:
        result = json.load(f)

    assert len(result) == 1
    assert result[0]["input"] == "Question: { Who was Albert Einstein? }"
    assert result[0]["output"] == "(JOIN fb\\:Albert_Einstein ...)"
    assert result[0]["history"] == []


def test_register_dataset(tmp_path, monkeypatch):
    path = tmp_path / "dataset_info.json"

    monkeypatch.setattr(
        prepare_llm_data,
        "DATASET_INFO_PATH",
        str(path),
    )

    prepare_llm_data.register_dataset("WebQSP", "train.jena")

    with open(path) as f:
        info = json.load(f)

    assert info["WebQSP_train.jena"] == {
        "file_name": "WebQSP_train.jena/examples.json",
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": "history",
        },
    }