import json
import pytest

import src.generate_predictions as generate_predictions


def test_build_question():
    question = "Who was Albert Einstein?"

    assert generate_predictions.build_question(question) == (
        "Generate a Logical Form query that retrieves the information "
        "corresponding to the given question.\n\n"
        "Question: { Who was Albert Einstein? }"
    )


def test_gold_rank():
    predictions = [
        "(JOIN fb:Albert_Einstein fbp:people.person.place_of_birth)",
        "(JOIN fb:Albert_Einstein fbp:people.person.sibling_s)",
    ]

    assert generate_predictions._gold_rank(
        predictions,
        "(JOIN fb:Albert_Einstein fbp:people.person.sibling_s)",
    ) == 1


def test_gold_rank_case_and_whitespace():
    predictions = [
        "  (JOIN fb:Albert_Einstein fbp:people.person.place_of_birth)  "
    ]

    assert generate_predictions._gold_rank(
        predictions,
        "(join fb:albert_einstein fbp:people.person.place_of_birth)",
    ) == 0


def test_gold_rank_not_found():
    predictions = [
        "(JOIN fb:Albert_Einstein fbp:people.person.place_of_birth)",
    ]

    assert generate_predictions._gold_rank(
        predictions,
        "(JOIN fb:Albert_Einstein fbp:people.person.sibling_s)",
    ) is None


def test_load_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.jsonl"

    records = [
        {"idx": 0, "item": {"id": "first"}},
        {"idx": 1, "item": {"id": "second"}},
    ]

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    assert generate_predictions.load_checkpoint(str(path)) == {
        0: {"id": "first"},
        1: {"id": "second"},
    }


def test_load_checkpoint_missing_file(tmp_path):
    assert generate_predictions.load_checkpoint(
        str(tmp_path / "missing.jsonl")
    ) == {}


def test_append_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.jsonl"

    generate_predictions.append_checkpoint(
        str(path),
        3,
        {"question": "Example"},
    )

    with open(path, encoding="utf-8") as f:
        record = json.loads(f.readline())

    assert record == {
        "idx": 3,
        "item": {"question": "Example"},
    }


def test_run_manifest_dict():
    args = type(
        "Args",
        (),
        {
            "dataset": "WebQSP",
            "split": "test",
            "mode": "sparql",
            "num_beams": 8,
            "max_new_tokens": 512,
            "diversity_penalty": 0.5,
            "oracle": False,
        },
    )()

    assert generate_predictions._run_manifest_dict(args) == {
        "dataset": "WebQSP",
        "split": "test",
        "mode": "sparql",
        "num_beams": 8,
        "max_new_tokens": 512,
        "diversity_penalty": 0.5,
        "oracle": False,
    }


def test_check_or_write_manifest(tmp_path):
    manifest = {
        "dataset": "WebQSP",
        "split": "test",
        "mode": "sparql",
        "num_beams": 8,
        "max_new_tokens": 512,
        "diversity_penalty": 0.5,
        "oracle": False,
    }

    generate_predictions._check_or_write_manifest(
        str(tmp_path),
        manifest,
    )

    path = tmp_path / "run_manifest.json"

    with open(path, encoding="utf-8") as f:
        assert json.load(f) == manifest


def test_check_or_write_manifest_rejects_different_manifest(tmp_path):
    manifest = {
        "dataset": "WebQSP",
        "split": "test",
        "mode": "sparql",
        "num_beams": 8,
        "max_new_tokens": 512,
        "diversity_penalty": 0.5,
        "oracle": False,
    }

    generate_predictions._check_or_write_manifest(
        str(tmp_path),
        manifest,
    )

    different = {**manifest, "num_beams": 4}

    with pytest.raises(ValueError, match="different parameters"):
        generate_predictions._check_or_write_manifest(
            str(tmp_path),
            different,
        )


def test_build_meta():
    args = type(
        "Args",
        (),
        {
            "dataset": "WebQSP",
            "split": "test",
            "mode": "sparql",
            "num_beams": 4,
            "max_new_tokens": 128,
            "diversity_penalty": 0.5,
            "oracle": False,
        },
    )()

    meta = generate_predictions._build_meta(
        args=args,
        model_id="test-model",
        num_items=3,
        beam_counts=[4, 2, 3],
        gold_in_beams=[True, True, False],
        gold_at_rank0=[True, False, False],
    )

    assert meta["dataset"] == "WebQSP"
    assert meta["model_id"] == "test-model"
    assert meta["num_items"] == 3
    assert meta["gold_in_beams_count"] == 2
    assert meta["gold_in_beams_pct"] == 66.67
    assert meta["gold_at_rank0_count"] == 1
    assert meta["gold_at_rank0_pct"] == 33.33
    assert meta["mean_beams_per_item"] == 3
    assert meta["median_beams_per_item"] == 3
    assert meta["min_beams"] == 2
    assert meta["max_beams"] == 4