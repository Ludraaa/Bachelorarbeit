import pytest

from src.kb.wikidata import Wikidata


@pytest.fixture
def kb():
    return Wikidata()


@pytest.mark.parametrize(
    "name, uri, expected",
    [
        (
            "Entity",
            "http://www.wikidata.org/entity/Q42",
            "http://www.wikidata.org/entity/Q42",
        ),
        (
            "Predicate",
            "http://www.wikidata.org/prop/direct/P31",
            "http://www.wikidata.org/entity/P31",
        ),
        (
            "Statement predicate",
            "http://www.wikidata.org/prop/statement/P31",
            "http://www.wikidata.org/entity/P31",
        ),
        (
            "Qualifier predicate",
            "http://www.wikidata.org/prop/qualifier/P580",
            "http://www.wikidata.org/entity/P580",
        ),
        (
            "Invalid URI",
            "http://example.org/something",
            None,
        ),
    ],
)
def test_normalize(name, uri, expected, kb):
    assert kb.normalize(uri) == expected


@pytest.mark.parametrize(
    "name, bindings, expected",
    [
        (
            "English found, other language found also",
            [
                {
                    "uri": {"value": "http://www.wikidata.org/entity/Q42"},
                    "label": {"value": "Douglas Adams", "xml:lang": "en"},
                },                
                {
                    "uri": {"value": "http://www.wikidata.org/entity/Q42"},
                    "label": {"value": "Mensch", "xml:lang": "de"},
                },
            ],
            {
                "http://www.wikidata.org/entity/Q42": "Douglas Adams",
            },
        ),
        (
            "No English candidate",
            [
                {
                    "uri": {"value": "http://www.wikidata.org/entity/Q42"},
                    "label": {"value": "Douglas Adams Mul", "xml:lang": "mul"},
                },
                {
                    "uri": {"value": "http://www.wikidata.org/entity/Q42"},
                    "label": {"value": "Douglas Adams De", "xml:lang": "de"},
                },
            ],
            {
                "http://www.wikidata.org/entity/Q42": "Douglas Adams Mul",
            },
        ),
        (
            "Entity ID is not used as label",
            [
                {
                    "uri": {"value": "http://www.wikidata.org/entity/Q42"},
                    "label": {"value": "Q42"},
                },
            ],
            {},
        ),
    ],
)
def test_parse_label_results(name, bindings, expected, kb):
    assert kb.parse_label_results(bindings) == expected


@pytest.mark.parametrize(
    "name, uri, label, expected",
    [
        (
            "Entity with label",
            "http://www.wikidata.org/entity/Q42",
            "Douglas Adams",
            "wd:Douglas_Adams",
        ),
        (
            "Direct predicate with label",
            "http://www.wikidata.org/prop/direct/P31",
            "instance of",
            "wdt:instance_of",
        ),
        (
            "Direct predicate without label",
            "http://www.wikidata.org/prop/direct/P31",
            "",
            "wdt:P31",
        ),
        (
            "Statement predicate",
            "http://www.wikidata.org/prop/statement/P31",
            "instance of",
            "ps:instance_of",
        ),
        (
            "Qualifier predicate",
            "http://www.wikidata.org/prop/qualifier/P580",
            "start time",
            "pq:start_time",
        ),
    ],
)
def test_format_label(name, uri, label, expected, kb):
    assert kb.format_label(uri, label) == expected

@pytest.mark.parametrize(
    "name, prediction, entities, predicates",
    [
        (
            "Simple",
            "(JOIN wd:Douglas_Adams wdt:instance_of)",
            ["Douglas_Adams"],
            ["instance_of"],
        ),
        (
            "Duplicate entity",
            "(AND (JOIN wd:Douglas_Adams wdt:instance_of) "
            "(JOIN wd:Douglas_Adams wdt:occupation))",
            ["Douglas_Adams"],
            ["instance_of", "occupation"],
        ),
        (
            "Predicate label containing parentheses",
            "(JOIN wd:Something wdt:located_in_(on_physical_feature))",
            ["Something"],
            ["located_in_(on_physical_feature)"],
        ),
        (
            "Predicate label containing slash",
            "(JOIN wd:Something wdt:located_in/on_physical_feature)",
            ["Something"],
            ["located_in/on_physical_feature"],
        ),
        (
            "Predicate property path",
            "(JOIN wd:Something wdt:subclass_of/wdt:instance_of)",
            ["Something"],
            ["subclass_of", "instance_of"],
        ),
        (
            "Predicate inverse path",
            "(JOIN wd:Something ^wdt:instance_of)",
            ["Something"],
            ["instance_of"],
        ),
        (
            "Entity label containing parentheses",
            "(JOIN wd:Albert_(Einstein) wdt:instance_of)",
            ["Albert_(Einstein)"],
            ["instance_of"],
        ),
        (
            "Entity label with whitespace",
            "(JOIN wd:Albert Einstein wdt:instance_of)",
            ["Albert"],
            ["instance_of"],
        ),
    ],
)
def test_extract_from_prediction(
    name, prediction, entities, predicates, kb
):
    ent, pred = kb.extract_from_prediction(prediction)
    assert ent == entities
    assert pred == predicates
    

def test_substitute(kb):
    prediction = (
        "(JOIN wd:Douglas_Adams "
        "(R wdt:instance_of) "
        "wdt:occupation)"
    )

    result = kb.substitute(
        prediction,
        entity_map={"Douglas_Adams": "Q42"},
        predicate_map={
            "instance_of": "P31",
            "occupation": "P106",
        },
    )

    assert result == (
        "(JOIN <http://www.wikidata.org/entity/Q42> "
        "(R <http://www.wikidata.org/prop/direct/P31>) "
        "<http://www.wikidata.org/prop/direct/P106>)"
    )
    
def test_substitute_predicate_prefixes(kb):
    prediction = (
        "(AND wdt:instance_of "
        "p:instance_of "
        "pq:instance_of)"
    )

    result = kb.substitute(
        prediction,
        entity_map={},
        predicate_map={"instance_of": "P31"},
    )

    assert result == (
        "(AND <http://www.wikidata.org/prop/direct/P31> "
        "<http://www.wikidata.org/prop/P31> "
        "<http://www.wikidata.org/prop/qualifier/P31>)"
    )