import pytest

from src.sexpr.jena_interface import (
    remove_comments,
    _mask_literals,
    _collapse_whitespace_outside_literals,
    _plain_vars_at_depth0,
    _add_missing_group_by,
    fix_sparql_for_jena,
    detect_query_form,
    restore_query_form,
    strip_prefix_and_expand
)

def normalize(s):
    return " ".join(s.split())

@pytest.mark.parametrize(
    "name, query, expected",
    [
        (
            "removes comment",
            "SELECT ?x # comment\nWHERE { ?x ?p ?o }",
            "SELECT ?x \nWHERE { ?x ?p ?o }",
        ),
        (
            "keep hash in string",
            'SELECT ?x WHERE { ?x ?p "#not-a-comment" }',
            'SELECT ?x WHERE { ?x ?p "#not-a-comment" }',
        ),
        (
            "keep hash in IRI",
            "SELECT ?x WHERE { ?x ?p <http://example.org/#thing> }",
            "SELECT ?x WHERE { ?x ?p <http://example.org/#thing> }",
        ),
    ],
)
def test_remove_comments(name, query, expected):
    assert remove_comments(query) == expected
    
    
def test_mask_literals():
    query = 'SELECT ?x WHERE { ?x ?p "hello" . }'
    masked = _mask_literals(query)

    assert len(_mask_literals(query)) == len(query)
    assert "SELECT ?x WHERE {" in masked
    assert '"' in masked
    assert "hello" not in masked
    assert masked == 'SELECT ?x WHERE { ?x ?p "     " . }'


@pytest.mark.parametrize(
    "query, expected",
    [
        (
            "SELECT   ?x   WHERE   { ?x ?p ?o }",
            "SELECT ?x WHERE { ?x ?p ?o }",
        ),
        (
            'SELECT   "hello   world"   WHERE { }',
            'SELECT "hello   world" WHERE { }',
        )
    ],
)
def test_collapse_whitespace(query, expected):
    assert _collapse_whitespace_outside_literals(query) == expected
    

@pytest.mark.parametrize(
    "select_list, expected",
    [
        (
            "?person ?name",
            ["?person", "?name"],
        ),
        (
            "?person (COUNT(?friend) AS ?count)",
            ["?person"],
        ),
        (
            "(COUNT(?friend) AS ?count)",
            [],
        ),
        (
            "?person (COUNT(?friend) AS ?count) ?name",
            ["?person", "?name"],
        ),
        (
            "?x (SUM(?value) AS ?total) ?y",
            ["?x", "?y"],
        ),
        (
            "?x (COUNT(IF(?a, ?b, ?c)) AS ?count) ?y",
            ["?x", "?y"],
        ),
    ],
)
def test_plain_vars_at_depth0(select_list, expected):
    assert _plain_vars_at_depth0(select_list) == expected


def test_add_missing_group_by():
    query = """
        SELECT ?person (COUNT(?friend) AS ?count)
        WHERE {
            ?person ?p ?friend
        }
    """

    expected = """
        SELECT ?person (COUNT(?friend) AS ?count)
        {
            ?person ?p ?friend
        } GROUP BY ?person
    """

    assert _add_missing_group_by(query) == expected
    
    
def test_add_missing_group_by_does_not_duplicate_existing_group_by():
    query = """
        SELECT ?person (COUNT(?friend) AS ?count)
        WHERE {
            ?person ?p ?friend
        }
        GROUP BY ?person
    """

    result = _add_missing_group_by(query)

    assert normalize(result).count("GROUP BY") == 1
    

def test_add_missing_group_by_handles_nested_select():
    query = """
        SELECT ?person {
            {
                SELECT ?person (COUNT(?friend) AS ?count)
                WHERE {
                    ?person ?p ?friend
                }
            }
        }
    """

    result = _add_missing_group_by(query)

    assert "GROUP BY ?person" in normalize(result)
    

@pytest.mark.parametrize(
    "query, expected",
    [
        ("SELECT ?x WHERE { ?x ?p ?o }", "SELECT"),
        ("PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x WHERE { ?x ?p ?o }", "SELECT"),
        ("ASK WHERE { ?x ?p ?o }", "ASK"),
        ("ASK { ?x ?p ?o }", "ASK"),
    ],
)
def test_detect_query_form(query, expected):
    assert detect_query_form(query) == expected


def test_restore_ask_query():
    sparql = "SELECT ?x WHERE { ?x ?p ?o }"

    assert restore_query_form("ASK", sparql) == "ASK WHERE { ?x ?p ?o }"
    

def test_strip_prefix_and_expand():
    algebra = """
        (prefix ((wd <http://www.wikidata.org/entity/>))
            (bgp (triple ?x wd:Q42 ?y)))
    """

    result = strip_prefix_and_expand(
        algebra,
        {"wd": "http://www.wikidata.org/entity/"},
    )

    assert "wd:Q42" not in result
    assert "<http://www.wikidata.org/entity/Q42>" in result