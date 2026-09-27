import json
from types import SimpleNamespace
import pytest

import src.resolve_predictions as resolve_predictions
from src.linkers.base import LinkingOutput


@pytest.mark.parametrize(
    "name, value, fallback, expected",
    [
        ("single value", "5", 25, [5]),
        ("multiple values", "15, 5", 25, [15, 5]),
        ("whitespace", " 15,  5 ", 25, [15, 5]),
        ("empty", "", 25, [25]),
    ],
)
def test__parse_ints(name, value, fallback, expected):
    assert resolve_predictions._parse_ints(value, fallback) == expected


@pytest.mark.parametrize(
    "name, value, fallback, expected",
    [
        ("single value", "1.0", 0.0, [1.0]),
        ("multiple values", "1.0, 0.5", 0.0, [1.0, 0.5]),
        ("whitespace", " 1.0,  0.5 ", 0.0, [1.0, 0.5]),
        ("empty", "", 0.5, [0.5]),
    ],
)
def test__parse_floats(name, value, fallback, expected):
    assert resolve_predictions._parse_floats(value, fallback) == expected


@pytest.mark.parametrize(
    "name, values, pass_idx, expected",
    [
        ("first value", [10, 20], 0, 10),
        ("second value", [10, 20], 1, 20),
        ("reuse last value", [10, 20], 2, 20),
        ("reuse last value", [10, 20], 5, 20),
        ("single value", [10], 3, 10),
    ],
)
def test__get_pass_val(name, values, pass_idx, expected):
    assert resolve_predictions._get_pass_val(values, pass_idx) == expected


@pytest.mark.parametrize(
    "name, deadline, expected",
    [
        ("no deadline", None, False),
        ("expired deadline", 0.0, True),
    ],
)
def test__deadline_exceeded(name, deadline, expected):
    assert resolve_predictions._deadline_exceeded(deadline) is expected


@pytest.mark.parametrize(
    "name, item, expected",
    [
        ("non-empty answer", {"answer": [["x"]]}, True),
        ("empty list", {"answer": []}, False),
        ("empty string", {"answer": ""}, False),
        ("missing answer", {}, False),
        ("none", {"answer": None}, False),
    ],
)
def test__has_gold_answer(name, item, expected):
    assert resolve_predictions._has_gold_answer(item) is expected


def test_load_predictions(tmp_path):
    data_dir = tmp_path / "data"
    path = (data_dir / "WebQSP" / "predictions" / "test-model" / "run1" / "raw")
    path.mkdir(parents=True)

    (path / "WebQSP_test.sparql.json").write_text(
        json.dumps(
            {
                "meta": {"max_beams": 8},
                "items": [{"question": "Example"}],
            }
        ),
        encoding="utf-8",
    )

    items, meta = resolve_predictions.load_predictions(
        str(data_dir),
        "WebQSP",
        "test-model",
        "run1",
        "test",
        "sparql",
    )

    assert items == [{"question": "Example"}]
    assert meta == {"max_beams": 8}


def test_resolve_output_path(tmp_path):
    args = SimpleNamespace(
        data_dir=str(tmp_path),
        dataset="WebQSP",
        model_id="test-model",
        split="test",
        mode="sparql",
    )

    result = resolve_predictions.resolve_output_path(args, "run1")

    assert result == str(tmp_path / "WebQSP" / "predictions" / "test-model" / "run1" / "resolved" / "WebQSP_test.sparql.jsonl")


@pytest.mark.parametrize(
    "name, existing, expected_error",
    [
        ("new manifest", None, None),
        ("same manifest", {"kb": "Wikidata", "mode": "sparql"}, None),
        ("different manifest", {"kb": "Freebase", "mode": "sparql"}, ValueError),
    ],
)
def test__check_or_write_manifest(tmp_path, name, existing, expected_error):
    manifest = {"kb": "Wikidata", "mode": "sparql"}
    path = tmp_path / "run_manifest.json"

    if existing is not None:
        path.write_text(json.dumps(existing), encoding="utf-8")

    if expected_error is None:
        resolve_predictions._check_or_write_manifest(
            str(tmp_path),
            manifest,
        )

        assert json.loads(path.read_text(encoding="utf-8")) == manifest
    else:
        with pytest.raises(expected_error, match="different parameters"):
            resolve_predictions._check_or_write_manifest(
                str(tmp_path),
                manifest,
            )


@pytest.mark.parametrize(
    "name, content, expected_items, expected_count",
    [
        (
            "two valid lines",
            '{"id": 1}\n{"id": 2}\n',
            [{"id": 1}, {"id": 2}],
            2,
        ),
        (
            "empty lines are ignored",
            '{"id": 1}\n\n{"id": 2}\n',
            [{"id": 1}, {"id": 2}],
            2,
        ),
        (
            "invalid line is ignored",
            '{"id": 1}\ninvalid\n{"id": 2}\n',
            [{"id": 1}, {"id": 2}],
            2,
        ),
        (
            "empty file",
            "",
            [],
            0,
        ),
    ],
)
def test__load_existing_jsonl(
    tmp_path,
    name,
    content,
    expected_items,
    expected_count,
):
    path = tmp_path / "results.jsonl"
    path.write_text(content, encoding="utf-8")

    items, count = resolve_predictions._load_existing_jsonl(str(path))

    assert items == expected_items
    assert count == expected_count


def test__append_jsonl(tmp_path):
    path = tmp_path / "results.jsonl"

    resolve_predictions._append_jsonl(
        str(path),
        {"id": 1, "executable": True},
    )
    resolve_predictions._append_jsonl(
        str(path),
        {"id": 2, "executable": False},
    )

    assert [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ] == [
        {"id": 1, "executable": True},
        {"id": 2, "executable": False},
    ]


def test__finalize_to_json(tmp_path):
    jsonl_path = tmp_path / "results.jsonl"
    jsonl_path.write_text(
        '{"id": 1}\n{"id": 2}\n',
        encoding="utf-8",
    )

    result = resolve_predictions._finalize_to_json(
        str(jsonl_path),
        {"dataset": "WebQSP"},
    )

    assert result == str(tmp_path / "results.json")

    output = json.loads(
        (tmp_path / "results.json").read_text(encoding="utf-8")
    )

    assert output == {
        "meta": {"dataset": "WebQSP"},
        "items": [{"id": 1}, {"id": 2}],
    }


@pytest.mark.parametrize(
    "name, existing_files",
    [
        ("only final json", ["results.json"]),
        (
            "all files",
            [
                "results.json",
                "results.jsonl",
                "results.debug.json",
                "results.debug.jsonl",
            ],
        ),
    ],
)
def test__reset_if_already_finished(tmp_path, name, existing_files):
    for filename in existing_files:
        (tmp_path / filename).write_text("test", encoding="utf-8")

    resolve_predictions._reset_if_already_finished(
        str(tmp_path / "results.jsonl"),
        str(tmp_path / "results.json"),
        str(tmp_path / "results.debug.jsonl"),
        str(tmp_path / "results.debug.json"),
    )

    for filename in existing_files:
        assert not (tmp_path / filename).exists()


@pytest.mark.parametrize(
    "name, candidates, k, threshold, expected",
    [
        (
            "two entities",
            {
                "entity1": [("Q1", 1.0), ("Q2", 0.5)],
                "entity2": [("Q3", 0.8), ("Q4", 0.4)],
            },
            2,
            0.0,
            [
                ({"entity1": "Q1", "entity2": "Q3"}, 0.9),
                ({"entity1": "Q1", "entity2": "Q4"}, 0.7),
            ],
        ),
        (
            "unsorted candidates",
            {
                "entity": [("low", 0.2), ("high", 0.9)],
            },
            1,
            0.0,
            [({"entity": "high"}, 0.9)],
        ),
        (
            "empty candidates",
            {},
            5,
            0.0,
            [],
        ),
        (
            "empty candidate list",
            {"entity": []},
            5,
            0.0,
            [],
        ),
        (
            "threshold filters low score",
            {
                "entity": [("high", 0.9), ("low", 0.2)],
            },
            2,
            0.5,
            [({"entity": "high"}, 0.9)],
        ),
    ],
)
def test__kbest_cartesian(
    name,
    candidates,
    k,
    threshold,
    expected,
):
    result = resolve_predictions._kbest_cartesian(
        candidates,
        k,
        threshold,
    )

    assert len(result) == len(expected)

    for (actual_map, actual_score), (expected_map, expected_score) in zip(
        result,
        expected,
    ):
        assert actual_map == expected_map
        assert actual_score == pytest.approx(expected_score)


@pytest.mark.parametrize(
    "name, function, k, threshold",
    [
        ("entity permutation", "entity", 1, 0.0),
        ("relation permutation", "relation", 1, 0.0),
    ],
)
def test_permute_by_entity(name, function, k, threshold):
    candidates = {
        "x": [
            ("A", 1.0),
            ("B", 0.5),
        ]
    }

    if function == "entity":
        result = resolve_predictions.permute_by_entity(
            candidates,
            k,
            threshold,
        )
    else:
        result = resolve_predictions.permute_by_relation(
            candidates,
            k,
            threshold,
        )

    assert result == [({"x": "A"}, 1.0)]


@pytest.mark.parametrize(
    "name, query, mode, fallback, expected",
    [
        (
            "already SPARQL",
            "SELECT * WHERE { ?x ?p ?y }",
            "sparql",
            False,
            ["SELECT * WHERE { ?x ?p ?y }"],
        ),
        (
            "unknown mode",
            "some query",
            "unknown",
            False,
            ["some query"],
        ),
    ],
)
def test_to_sparql(name, query, mode, fallback, expected):
    assert resolve_predictions.to_sparql(
        query,
        mode,
        fallback,
    ) == expected


@pytest.mark.parametrize(
    "name, query, expected_prefixes",
    [
        (
            "wikidata prefixes",
            "SELECT * WHERE { wd:Q42 wdt:P31 ?x }",
            [
                "PREFIX wd: <http://www.wikidata.org/entity/>",
                "PREFIX wdt: <http://www.wikidata.org/prop/direct/>",
            ],
        ),
        (
            "no prefixes used",
            "SELECT * WHERE { ?x ?p ?y }",
            [],
        ),
    ],
)
def test_inject_prefixes(name, query, expected_prefixes):
    result = resolve_predictions.inject_prefixes(
        query,
        {
            "wd": "http://www.wikidata.org/entity/",
            "wdt": "http://www.wikidata.org/prop/direct/",
        },
    )

    for prefix in expected_prefixes:
        assert prefix in result


@pytest.mark.parametrize(
    "name, bindings, expected",
    [
        ("non-empty bindings", [{"x": 1}], True),
        ("empty bindings", [], False),
        ("none", None, False),
    ],
)
def test__has_results(name, bindings, expected):
    assert resolve_predictions._has_results(bindings) is expected

# ---------------------------------------------------------------------------
# Runtime aggregation


def test__new_runtime_agg():
    result = resolve_predictions._new_runtime_agg(
        ["linker_a", "linker_b"]
    )

    assert result["total_count"] == 0
    assert result["total_sec"] == 0.0
    assert set(result["by_resolution"]) == {
        "linker_a",
        "linker_b",
        "_unresolved",
        "_timeout",
        "_skipped_no_gold",
    }


@pytest.mark.parametrize(
    "name, runtime, winning_linker, timed_out, skipped, expected_key",
    [
        (
            "resolved",
            2.5,
            "linker_a",
            False,
            False,
            "linker_a",
        ),
        (
            "unresolved",
            3.0,
            None,
            False,
            False,
            "_unresolved",
        ),
        (
            "timeout",
            1.5,
            None,
            True,
            False,
            "_timeout",
        ),
        (
            "skipped",
            0.0,
            None,
            False,
            True,
            "_skipped_no_gold",
        ),
    ],
)
def test__record_runtime(
    name,
    runtime,
    winning_linker,
    timed_out,
    skipped,
    expected_key,
):
    agg = resolve_predictions._new_runtime_agg(["linker_a"])

    resolve_predictions._record_runtime(
        agg,
        runtime,
        winning_linker,
        timed_out=timed_out,
        skipped=skipped,
    )

    assert agg["total_count"] == 1
    assert agg["total_sec"] == runtime
    assert agg["by_resolution"][expected_key]["count"] == 1
    assert agg["by_resolution"][expected_key]["total_sec"] == runtime


@pytest.mark.parametrize(
    "name, runtimes, expected_avg, expected_total",
    [
        ("two items", [1.0, 3.0], 2.0, 4.0),
        ("one item", [2.5], 2.5, 2.5),
        ("no items", [], None, 0.0),
    ],
)
def test__runtime_summary(
    name,
    runtimes,
    expected_avg,
    expected_total,
):
    agg = resolve_predictions._new_runtime_agg(["linker_a"])

    for runtime in runtimes:
        resolve_predictions._record_runtime(
            agg,
            runtime,
            "linker_a",
        )

    result = resolve_predictions._runtime_summary(agg)

    assert result["avg_runtime_sec_per_item"] == expected_avg
    assert result["total_runtime_sec"] == expected_total


def test__run_manifest_dict():
    args = SimpleNamespace(
        kb="Wikidata",
        mode="sparql",
        label_fallback=False,
        item_time_limit_sec=120.0,
    )

    result = resolve_predictions._run_manifest_dict(
        args=args,
        entity_linker_ids=["entity_a"],
        predicate_linker_ids=["predicate_a", "predicate_b"],
        linker_params={"entity_a": {"threshold": 0.5}},
        beam_limits=[8],
        k1_list=[10],
        t1_list=[0.2],
        k2_list=[5],
        t2_list=[0.1],
        n_passes=2,
    )

    assert result == {
        "kb": "Wikidata",
        "mode": "sparql",
        "entity_linkers": ["entity_a"],
        "predicate_linkers": ["predicate_a", "predicate_b"],
        "linker_params": {"entity_a": {"threshold": 0.5}},
        "beam_limits": [8, 8],
        "k1_per_pass": [10, 10],
        "t1_per_pass": [0.2, 0.2],
        "k2_per_pass": [5, 5],
        "t2_per_pass": [0.1, 0.1],
        "label_fallback": False,
        "item_time_limit_sec": 120.0,
    }


def test__build_meta():
    args = SimpleNamespace(
        dataset="WebQSP",
        split="test",
        model_id="test-model",
        kb="Wikidata",
        mode="sparql",
        item_time_limit_sec=60.0,
        data_dir="data",
        note="test run",
    )

    runtime_agg = resolve_predictions._new_runtime_agg(
        ["predicate_a"]
    )

    resolve_predictions._record_runtime(
        runtime_agg,
        2.0,
        "predicate_a",
    )

    result = resolve_predictions._build_meta(
        args=args,
        entity_linker_ids=["entity_a"],
        predicate_linker_ids=["predicate_a"],
        beam_limits=[8],
        k1_list=[10],
        t1_list=[0.2],
        k2_list=[5],
        t2_list=[0.1],
        num_items=2,
        executable_count=1,
        timeout_count=0,
        skipped_count=1,
        pass_counts={"predicate_a": 1},
        entity_linker_params={"entity_a": {}},
        predicate_linker_params={"predicate_a": {}},
        runtime_agg=runtime_agg,
        label_fallback=False,
        run_config_name="run1",
        linker_combo_id="entity_a+predicate_a",
    )

    assert result["dataset"] == "WebQSP"
    assert result["model_id"] == "test-model"
    assert result["kb"] == "Wikidata"
    assert result["num_items"] == 2
    assert result["num_executable"] == 1
    assert result["executable_pct"] == 50.0
    assert result["num_skipped_no_gold"] == 1
    assert result["skipped_no_gold_pct"] == 50.0
    assert result["pass_counts"] == {"predicate_a": 1}
    assert result["runtime"]["total_runtime_sec"] == 2.0