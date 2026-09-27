import re
import pytest

from src.kb.base import BaseKB


class TestKB(BaseKB):
    LABEL_QUERY = "SELECT * WHERE { VALUES ?x { {values} } }"
    ENTITY_PATTERN = re.compile(r"<(http://example\.org/entity/[^>]+)>")
    RELATION_PATTERN = re.compile(r"<(http://example\.org/prop/[^>]+)>")
    ANSWER_URI_PATTERNS = [
        re.compile(r"http://example\.org/entity/(.+)")
    ]

    @staticmethod
    def normalize(uri):
        return uri

    @staticmethod
    def parse_label_results(bindings):
        return {}

    @staticmethod
    def format_label(uri, label):
        return label

    @staticmethod
    def extract_from_prediction(prediction):
        return [], []

    @staticmethod
    def substitute(
        prediction, entity_map, predicate_map, expand_uris=True
    ):
        return prediction


@pytest.fixture
def kb():
    return TestKB()


@pytest.mark.parametrize(
    "name, sexpr, expected",
    [
        (
            "single entity",
            "(JOIN <http://example.org/entity/Q42>)",
            ["http://example.org/entity/Q42"],
        ),
        (
            "multiple entities",
            "(AND <http://example.org/entity/Q42> "
            "<http://example.org/entity/Q43>)",
            [
                "http://example.org/entity/Q42",
                "http://example.org/entity/Q43",
            ],
        ),
        (
            "duplicate entity",
            "(AND <http://example.org/entity/Q42> "
            "<http://example.org/entity/Q42>)",
            ["http://example.org/entity/Q42"],
        ),
        (
            "no entities",
            "(JOIN <http://example.org/prop/P31>)",
            [],
        ),
    ],
)
def test_extract_entities(name, sexpr, expected, kb):
    assert sorted(kb.extract_entities(sexpr)) == sorted(expected)


@pytest.mark.parametrize(
    "name, sexpr, expected",
    [
        (
            "single relation",
            "(JOIN <http://example.org/prop/P31>)",
            ["http://example.org/prop/P31"],
        ),
        (
            "multiple relations",
            "(JOIN <http://example.org/prop/P31> "
            "<http://example.org/prop/P106>)",
            [
                "http://example.org/prop/P31",
                "http://example.org/prop/P106",
            ],
        ),
        (
            "duplicate relation",
            "(AND <http://example.org/prop/P31> "
            "<http://example.org/prop/P31>)",
            ["http://example.org/prop/P31"],
        ),
        (
            "no relations",
            "(JOIN <http://example.org/entity/Q42>)",
            [],
        ),
    ],
)
def test_extract_relations(name, sexpr, expected, kb):
    assert sorted(kb.extract_relations(sexpr)) == sorted(expected)


@pytest.mark.parametrize(
    "name, uri, expected",
    [
        (
            "matching custom pattern",
            "http://example.org/entity/Q42",
            "Q42",
        ),
        (
            "second matching pattern",
            "http://example.org/other/Q42",
            "Q42",
        ),
        (
            "fallback to last slash",
            "http://other.example.org/entity/Q42",
            "Q42",
        ),
        (
            "no slash",
            "Q42",
            "Q42",
        ),
    ],
)
def test_normalize_answer_uri(name, uri, expected, kb):
    assert kb.normalize_answer_uri(uri) == expected


@pytest.mark.parametrize(
    "name, bindings, expected",
    [
        (
            "single type",
            [
                {
                    "uri": {
                        "value": "http://example.org/entity/Q42"
                    }
                }
            ],
            {"http://example.org/entity/Q42"},
        ),
        (
            "multiple types",
            [
                {
                    "uri": {
                        "value": "http://example.org/entity/Q42"
                    }
                },
                {
                    "uri": {
                        "value": "http://example.org/entity/Q43"
                    }
                },
            ],
            {
                "http://example.org/entity/Q42",
                "http://example.org/entity/Q43",
            },
        ),
        (
            "missing uri ignored",
            [
                {
                    "label": {
                        "value": "Something"
                    }
                }
            ],
            set(),
        ),
    ],
)
def test_parse_type_results(name, bindings, expected, kb):
    assert kb.parse_type_results(bindings) == expected


@pytest.mark.parametrize(
    "name, uri, expected",
    [
        (
            "rdf prefix",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "rdf:type",
        ),
        (
            "rdfs prefix",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "rdfs:label",
        ),
        (
            "custom KB prefix",
            "http://example.org/entity/Q42",
            "test:Q42",
        ),
        (
            "unknown prefix",
            "http://unknown.example.org/Q42",
            "",
        ),
        (
            "prefix without local name",
            "http://example.org/entity/",
            "",
        ),
    ],
)
def test__format_via_common_prefixes(name, uri, expected):
    class PrefixKB(TestKB):
        KB_PREFIXES = {
            "test": "http://example.org/entity/",
        }

    kb = PrefixKB()

    assert kb._format_via_common_prefixes(uri) == expected


@pytest.mark.parametrize(
    "name, uri, label, expected",
    [
        (
            "returns label",
            "http://example.org/prop/P31",
            "instance of",
            "instance of",
        ),
        (
            "empty label",
            "http://example.org/prop/P31",
            "",
            "",
        ),
    ],
)
def test_format_relation_label(name, uri, label, expected, kb):
    assert kb.format_relation_label(uri, label) == expected