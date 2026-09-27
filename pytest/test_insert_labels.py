from src.insert_labels import (
    discover_paths,
    load_cache,
    save_cache,
)

def test_load_cache_missing(tmp_path):
    path = tmp_path / "missing.json"

    assert load_cache(path) == {}


def test_save_load_cache_roundtrip(tmp_path):
    path = tmp_path / "cache" / "labels.json"

    cache = {
        "http://example.org/Q1": "Alice",
        "http://example.org/Q2": None,
    }

    save_cache(cache, path)

    assert path.exists()
    assert load_cache(path) == cache


def test_discover_paths(tmp_path, monkeypatch):

    sexpr = tmp_path / "WebQSP" / "sexpr"
    sexpr.mkdir(parents=True)

    (sexpr / "WebQSP_train.sparql.expr.json").touch()
    (sexpr / "WebQSP_train.grisp.expr.json").touch()
    (sexpr / "WebQSP_dev.sparql.expr.json").touch()
    (sexpr / "WebQSP_test.sparql.expr.json").touch()

    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    result = discover_paths("WebQSP")

    assert set(result.keys()) == {"train", "dev", "test"}

    assert set(result["train"].keys()) == {
        "sparql",
        "grisp",
    }

    assert (
        result["train"]["sparql"]["merged"]
        == tmp_path
        / "WebQSP"
        / "generation"
        / "merged"
        / "WebQSP_train.sparql.json"
    )
