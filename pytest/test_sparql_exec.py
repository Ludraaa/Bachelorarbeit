import pytest

from src.utils import sparql_exec


class TestKB:
    @staticmethod
    def normalize_answer_uri(uri):
        return uri.rsplit("/", 1)[-1]


@pytest.fixture
def kb():
    return TestKB()


@pytest.mark.parametrize(
    "name, results, expected",
    [
        (
            "ask true",
            True,
            [["true"]],
        ),
        (
            "ask false",
            False,
            [["false"]],
        ),
        (
            "empty select",
            [],
            [],
        ),
        (
            "invalid result type",
            None,
            [],
        ),
        (
            "uri",
            [
                {
                    "x": {
                        "type": "uri",
                        "value": "http://example.org/Q42",
                    }
                }
            ],
            [["Q42"]],
        ),
        (
            "english literal",
            [
                {
                    "x": {
                        "type": "literal",
                        "value": "Hello",
                        "xml:lang": "en",
                    }
                }
            ],
            [["hello"]],
        ),
        (
            "literal without language",
            [
                {
                    "x": {
                        "type": "literal",
                        "value": "Hello",
                    }
                }
            ],
            [["hello"]],
        ),
        (
            "non english literal skipped",
            [
                {
                    "x": {
                        "type": "literal",
                        "value": "Hallo",
                        "xml:lang": "de",
                    }
                }
            ],
            [],
        ),
        (
            "empty literal skipped",
            [
                {
                    "x": {
                        "type": "literal",
                        "value": "   ",
                    }
                }
            ],
            [],
        ),
        (
            "invalid row skipped",
            ["not a row"],
            [],
        ),
    ],
)
def test_bindings_to_rows(name, results, expected, kb):
    assert sparql_exec.bindings_to_rows(results, kb) == expected


@pytest.mark.parametrize(
    "name, answer, expected",
    [
        (
            "empty answer",
            [],
            [],
        ),
        (
            "non list",
            None,
            [],
        ),
        (
            "flat answer",
            ["Q42", "Q43"],
            [["Q42"], ["Q43"]],
        ),
        (
            "already nested",
            [["Q42"], ["Q43"]],
            [["Q42"], ["Q43"]],
        ),
        (
            "multiple values per row",
            [["Q42", "Q43"], ["Q44"]],
            [["Q42", "Q43"], ["Q44"]],
        ),
    ],
)
def test_ensure_rows(name, answer, expected):
    assert sparql_exec.ensure_rows(answer) == expected