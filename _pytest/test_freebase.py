import pytest

from src.kb.freebase import Freebase


@pytest.fixture
def kb():
    return Freebase()


@pytest.mark.parametrize(
    "name, uri, expected",
    [
        (
            "Entity",
            "http://rdf.freebase.com/ns/m.0f8l9c",
            "http://rdf.freebase.com/ns/m.0f8l9c",
        ),
        (
            "Predicate",
            "http://rdf.freebase.com/ns/people.person.place_of_birth",
            None
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
                    "uri": {"value": "http://rdf.freebase.com/ns/m.123"},
                    "label": {"value": "Mensch", "xml:lang": "de"},
                },
                {
                    "uri": {"value": "http://rdf.freebase.com/ns/m.123"},
                    "label": {"value": "Human", "xml:lang": "en"},
                },
            ],
            {"http://rdf.freebase.com/ns/m.123": "Human"},
        ),
        (
            "No English candidate",
            [
                {
                    "uri": {"value": "http://rdf.freebase.com/ns/m.123"},
                    "label": {"value": "인간", "xml:lang": "kr"},
                },
                {
                    "uri": {"value": "http://rdf.freebase.com/ns/m.123"},
                    "label": {"value": "Mensch", "xml:lang": "de"},
                },
            ],
            {"http://rdf.freebase.com/ns/m.123": "인간"},
        )
    ]
)
def test_parse_label_results(name, bindings, expected, kb):
    assert kb.parse_label_results(bindings) == expected


@pytest.mark.parametrize(
    "name, uri, label, expected",
    [
        (
            "Entity + label",
            "http://rdf.freebase.com/ns/m.0f8l9c",
            "SomeLabel",
            "fb:SomeLabel"
        ),
        (
            "Entity + empty label",
            "http://rdf.freebase.com/ns/m.0f8l9c",
            "",
            "fb:m.0f8l9c"
        ),
        (
            "Entity + no label",
            "http://rdf.freebase.com/ns/m.0f8l9c",
            None,
            "fb:m.0f8l9c"
        ),
        (
            "Predicate + no label",
            "http://rdf.freebase.com/ns/people.person.place_of_birth",
            None,
            "fbp:people.person.place_of_birth"
        ),
        (
            "Predicate + label",
            "http://rdf.freebase.com/ns/people.person.place_of_birth",
            "SomeLabel",
            "fbp:people.person.place_of_birth"
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
            "(JOIN fb:Albert_Einstein fbp:people.person.place_of_birth)",
            ["Albert_Einstein"],
            ["people.person.place_of_birth"]
        ),
        (
            "Duplicate entity",
            "(AND (JOIN fb:Albert_Einstein fbp:people.person.place_of_birth) (JOIN fb:Albert_Einstein fbp:people.person.sibling_s)",
            ["Albert_Einstein"],
            ["people.person.place_of_birth", "people.person.sibling_s"]
        ),
        (
            "Parenthesis",
            "(JOIN fb:Albert(name_(german))_Einstein fbp:people.person(and_aliens(())).place_of_birth)",
            ["Albert(name_(german))_Einstein"],
            ["people.person(and_aliens(())).place_of_birth"]
        ),
        (
            "Bad format",
            "(JOIN fb:Albert_Einstein fbp:people.person.sibling)_s)",
            ["Albert_Einstein"],
            ["people.person.sibling"]
        ),
        (
            "No underscore",
            "(JOIN fb:Albert Einstein fbp:people.person.sibling_s)",
            ["Albert"],
            ["people.person.sibling_s"]
        ),
    ],
)
def test_extract_from_prediction(name, prediction, entities, predicates, kb):
    ent, pred = kb.extract_from_prediction(prediction)
    assert ent == entities
    assert pred == predicates


def test_substitute(kb):
    prediction = (
        "(JOIN fb:Albert_Einstein "
        "(R rdfs:label) "
        "fbp:people.person.place_of_birth)"
    )

    result = kb.substitute(
        prediction,
        entity_map={"Albert_Einstein": "m.0f8l9c"},
        predicate_map={
            "people.person.place_of_birth": "people.person.place_of_birth"
        },
    )

    assert result == (
        "(JOIN <http://rdf.freebase.com/ns/m.0f8l9c> "
        "(R <http://www.w3.org/2000/01/rdf-schema#label>) "
        "<http://rdf.freebase.com/ns/people.person.place_of_birth>)"
    )