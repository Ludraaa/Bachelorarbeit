import json
import pytest

from src import eval_predictions


@pytest.mark.parametrize(
    "name, data_dir, dataset, model_id, run_stem, split, mode, expected",
    [
        (
            "default path",
            "data",
            "WebQSP",
            "model",
            "run1",
            "test",
            "sparql",
            "data/WebQSP/predictions/model/run1/resolved/WebQSP_test.sparql.json",
        ),
        (
            "different dataset and mode",
            "/tmp/data",
            "CWQ",
            "qwen",
            "experiment",
            "dev",
            "jena",
            "/tmp/data/CWQ/predictions/qwen/experiment/resolved/CWQ_dev.jena.json",
        ),
    ],
)
def test_resolved_path(
    name,
    data_dir,
    dataset,
    model_id,
    run_stem,
    split,
    mode,
    expected,
):
    assert str(
        eval_predictions.resolved_path(
            data_dir,
            dataset,
            model_id,
            run_stem,
            split,
            mode,
        )
    ) == expected


@pytest.mark.parametrize(
    "name, data_dir, dataset, model_id, run_stem, split, mode, expected",
    [
        (
            "default path",
            "data",
            "WebQSP",
            "model",
            "run1",
            "test",
            "sparql",
            "data/WebQSP/predictions/model/run1/evaluated/WebQSP_test.sparql.json",
        ),
        (
            "different dataset and mode",
            "/tmp/data",
            "CWQ",
            "qwen",
            "experiment",
            "dev",
            "jena",
            "/tmp/data/CWQ/predictions/qwen/experiment/evaluated/CWQ_dev.jena.json",
        ),
    ],
)
def test_evaluated_path(
    name,
    data_dir,
    dataset,
    model_id,
    run_stem,
    split,
    mode,
    expected,
):
    assert str(
        eval_predictions.evaluated_path(
            data_dir,
            dataset,
            model_id,
            run_stem,
            split,
            mode,
        )
    ) == expected


@pytest.mark.parametrize(
    "name, filename, expected",
    [
        (
            "normal result",
            "result.json",
            "result.analysis.json",
        ),
        (
            "dataset result",
            "WebQSP_test.sparql.json",
            "WebQSP_test.analysis.json",
        ),
    ],
)
def test_analysis_json_path(tmp_path, name, filename, expected):
    path = tmp_path / filename

    assert eval_predictions.analysis_json_path(path) == tmp_path / expected


@pytest.mark.parametrize(
    "name, filename, expected",
    [
        (
            "normal result",
            "result.json",
            "result_plots",
        ),
        (
            "dataset result",
            "WebQSP_test.sparql.json",
            "WebQSP_test.sparql_plots",
        ),
    ],
)
def test_analysis_plots_dir(tmp_path, name, filename, expected):
    path = tmp_path / filename

    assert eval_predictions.analysis_plots_dir(path) == tmp_path / expected


@pytest.mark.parametrize(
    "name, pred, gold, expected",
    [
        (
            "perfect match",
            [["a", "b"], ["c"]],
            [["a", "b"], ["c"]],
            {"exact_match": 1, "assignment_f1": 1.0, "hit1": 1},
        ),
        (
            "row order differs",
            [["c"], ["a", "b"]],
            [["a", "b"], ["c"]],
            {"exact_match": 1, "assignment_f1": 1.0, "hit1": 1},
        ),
        (
            "partial match",
            [["a"]],
            [["a", "b"]],
            {"exact_match": 0, "assignment_f1": 0.6667, "hit1": 1},
        ),
        (
            "no overlap",
            [["a"]],
            [["b"]],
            {"exact_match": 0, "assignment_f1": 0.0, "hit1": 0},
        ),
        (
            "empty prediction",
            [],
            [["a"]],
            {"exact_match": 0, "assignment_f1": 0.0, "hit1": 0},
        ),
        (
            "empty gold",
            [["a"]],
            [],
            {"exact_match": 0, "assignment_f1": 0.0, "hit1": 0},
        ),
    ],
)
def test_score(name, pred, gold, expected):
    assert eval_predictions.score(pred, gold) == expected


@pytest.mark.parametrize(
    "name, item, get_live_gold, live_only, expected, expected_note",
    [
        (
            "saved answers",
            {"answer": [["Q1"]]},
            False,
            False,
            [["Q1"]],
            "saved",
        ),
        (
            "strict live-only without live execution",
            {"answer": [["Q1"]]},
            False,
            True,
            [],
            "empty",
        ),
        (
            "missing saved answers",
            {},
            False,
            False,
            [],
            "saved",
        ),
        (
            "live requested but no SPARQL",
            {"answer": [["Q1"]]},
            True,
            False,
            [["Q1"]],
            "saved",
        ),
    ],
)
def test_get_gold_answers(
    name,
    item,
    get_live_gold,
    live_only,
    expected,
    expected_note,
):
    assert eval_predictions.get_gold_answers(
        item,
        "http://example.org/sparql",
        10,
        get_live_gold,
        live_only,
        {},
        None,
    ) == (expected, expected_note)


@pytest.mark.parametrize(
    "name, content, create_file, expected",
    [
        (
            "existing ledger",
            [{"run": 1}],
            True,
            [{"run": 1}],
        ),
        (
            "multiple entries",
            [{"run": 1}, {"run": 2}],
            True,
            [{"run": 1}, {"run": 2}],
        ),
        (
            "missing ledger",
            None,
            False,
            [],
        ),
    ],
)
def test_load_ledger(tmp_path, name, content, create_file, expected):
    path = tmp_path / "results.json"

    if create_file:
        path.write_text(json.dumps(content), encoding="utf-8")

    assert eval_predictions.load_ledger(str(path)) == expected


def test_save_ledger(tmp_path):
    path = tmp_path / "nested" / "results.json"
    ledger = [
        {"dataset": "WebQSP"},
        {"dataset": "CWQ"},
    ]

    eval_predictions.save_ledger(ledger, str(path))

    assert json.loads(path.read_text(encoding="utf-8")) == ledger


@pytest.mark.parametrize(
    "name, values, expected",
    [
        (
            "empty",
            [],
            {"count": 0},
        ),
        (
            "single value",
            [3],
            {
                "count": 1,
                "min": 3,
                "max": 3,
                "mean": 3.0,
                "median": 3.0,
                "p90": 3.0,
                "p95": 3.0,
                "p99": 3.0,
            },
        ),
        (
            "multiple values",
            [1, 2, 3, 4, 5],
            {
                "count": 5,
                "min": 1,
                "max": 5,
                "mean": 3.0,
                "median": 3.0,
                "p90": 4.6,
                "p95": 4.8,
                "p99": 4.96,
            },
        ),
    ],
)
def test__idx_stats(name, values, expected):
    assert eval_predictions._idx_stats(values) == expected


@pytest.mark.parametrize(
    "name, values, expected",
    [
        (
            "empty",
            [],
            {"count": 0},
        ),
        (
            "single value",
            [2.5],
            {
                "count": 1,
                "mean": 2.5,
                "median": 2.5,
                "std": 0.0,
                "min": 2.5,
                "max": 2.5,
            },
        ),
        (
            "multiple values",
            [1.0, 2.0, 3.0],
            {
                "count": 3,
                "mean": 2.0,
                "median": 2.0,
                "std": 0.8165,
                "min": 1.0,
                "max": 3.0,
            },
        ),
    ],
)
def test__float_stats(name, values, expected):
    assert eval_predictions._float_stats(values) == expected


@pytest.mark.parametrize(
    "name, item, expected",
    [
        (
            "winning linker",
            {"winning_pass_linker": "linker_a"},
            "linker_a",
        ),
        (
            "stale",
            {"stale": True, "winning_pass_linker": "linker_a"},
            "_stale",
        ),
        (
            "unresolved",
            {},
            "_unresolved",
        ),
    ],
)
def test__winning_pass_bucket(name, item, expected):
    assert eval_predictions._winning_pass_bucket(item) == expected


@pytest.mark.parametrize(
    "name, items, expected",
    [
        (
            "resolved and unresolved",
            [
                {"winning_pass_linker": "linker_a"},
                {"winning_pass_linker": "linker_a"},
                {},
            ],
            {
                "linker_a": {"count": 2, "pct": 66.67},
                "_unresolved": {"count": 1, "pct": 33.33},
            },
        ),
        (
            "stale",
            [{"stale": True}],
            {"_stale": {"count": 1, "pct": 100.0}},
        ),
        (
            "empty",
            [],
            {},
        ),
    ],
)
def test_build_distribution_analysis(name, items, expected):
    result = eval_predictions.build_distribution_analysis(
        items,
        ["linker_a"],
    )

    assert result["winning_pass"] == expected


@pytest.mark.parametrize(
    "name, items, expected",
    [
        (
            "empty",
            [],
            {},
        ),
        (
            "single status",
            [{"exec_status": "ok"}],
            {"ok": {"count": 1, "pct": 100.0}},
        ),
        (
            "missing status",
            [{}],
            {"no_query": {"count": 1, "pct": 100.0}},
        ),
    ],
)
def test__exec_status_breakdown(name, items, expected):
    assert eval_predictions._exec_status_breakdown(items) == expected


@pytest.mark.parametrize(
    "name, items, expected_key",
    [
        (
            "resolved",
            [
                {
                    "winning_pass_linker": "linker_a",
                    "executable": True,
                    "exec_status": "ok",
                    "exact_match": 1,
                    "hit1": 1,
                    "assignment_f1": 1.0,
                }
            ],
            "linker_a",
        ),
        (
            "unresolved",
            [
                {
                    "executable": False,
                    "exec_status": "no_query",
                    "exact_match": 0,
                    "hit1": 0,
                    "assignment_f1": 0.0,
                }
            ],
            "_unresolved",
        ),
        (
            "stale",
            [{"stale": True}],
            "_stale",
        ),
    ],
)
def test_build_per_linker_performance(name, items, expected_key):
    result = eval_predictions.build_per_linker_performance(
        items,
        ["linker_a"],
    )

    assert expected_key in result


@pytest.mark.parametrize(
    "name, items, idx_key, expected",
    [
        (
            "ranks grouped",
            [
                {"executed_beam_rank": 2, "assignment_f1": 0.5},
                {"executed_beam_rank": 2, "assignment_f1": 0.3},
                {"executed_beam_rank": 3, "assignment_f1": 0.7},
            ],
            "executed_beam_rank",
            {
                2: [0.5, 0.3],
                3: [0.7],
            },
        ),
    ],
)
def test__grouped_losses(name, items, idx_key, expected):
    assert eval_predictions._grouped_losses(items, idx_key) == expected


@pytest.mark.parametrize(
    "name, items, idx_key, original_cap, n_total, expected_loss",
    [
        (
            "no losses",
            [{"executed_beam_rank": 1, "assignment_f1": 1.0}],
            "executed_beam_rank",
            10,
            1,
            0.0,
        ),
        (
            "loss present",
            [{"executed_beam_rank": 2, "assignment_f1": 0.5}],
            "executed_beam_rank",
            10,
            10,
            0.0,
        ),
    ],
)
def test__param_sensitivity(
    name,
    items,
    idx_key,
    original_cap,
    n_total,
    expected_loss,
):
    result = eval_predictions._param_sensitivity(
        items,
        idx_key,
        original_cap,
        n_total,
    )

    assert set(result) == set(eval_predictions._F1_BUDGETS)

    if expected_loss == 0.0:
        assert all(value["f1_loss"] == 0.0 for value in result.values())


@pytest.mark.parametrize(
    "name, items, idx_key, original_cap, n_total, expected",
    [
        (
            "no data",
            [],
            "executed_beam_rank",
            10,
            1,
            {
                "caps": [10],
                "cum_loss_pct": [0.0],
            },
        ),
        (
            "one rank",
            [{"executed_beam_rank": 2, "assignment_f1": 0.5}],
            "executed_beam_rank",
            10,
            1,
            {
                "caps": [10, 2],
                "cum_loss_pct": [0.0, 50.0],
            },
        ),
    ],
)
def test__full_sensitivity_curve(
    name,
    items,
    idx_key,
    original_cap,
    n_total,
    expected,
):
    assert eval_predictions._full_sensitivity_curve(
        items,
        idx_key,
        original_cap,
        n_total,
    ) == expected


@pytest.mark.parametrize(
    "name, items, beam, k1, k2, n_total, expected",
    [
        (
            "nothing dropped",
            [
                {
                    "executed_beam_rank": 0,
                    "winning_entity_perm_idx": 0,
                    "winning_predicate_perm_idx": 0,
                    "assignment_f1": 1.0,
                }
            ],
            1,
            1,
            1,
            1,
            {
                "items_dropped": 0,
                "f1_loss": 0.0,
            },
        ),
        (
            "beam rank exceeds cap",
            [
                {
                    "executed_beam_rank": 2,
                    "winning_entity_perm_idx": 0,
                    "winning_predicate_perm_idx": 0,
                    "assignment_f1": 0.5,
                }
            ],
            1,
            10,
            10,
            1,
            {
                "items_dropped": 1,
                "f1_loss": 0.5,
            },
        ),
    ],
)
def test__combined_effect(
    name,
    items,
    beam,
    k1,
    k2,
    n_total,
    expected,
):
    assert eval_predictions._combined_effect(
        items,
        beam,
        k1,
        k2,
        n_total,
    ) == expected


@pytest.mark.parametrize(
    "name, lid, expected",
    [
        (
            "stale",
            "_stale",
            "Stale",
        ),
        (
            "unresolved",
            "_unresolved",
            "Unresolved",
        ),
        (
            "normal linker",
            "ChatKBQA.simple",
            "ChatKBQA.simple",
        ),
    ],
)
def test__display_label(name, lid, expected):
    assert eval_predictions._display_label(lid) == expected