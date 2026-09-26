import re
import requests

from src.utils.retry import call_with_retry
from src.sexpr.jena_interface import (
    fix_sparql_for_jena,
    sparql_to_algebra,
    algebra_to_sparql,
    strip_prefix_and_expand,
    detect_query_form,
    restore_query_form,
)

_SPARQL_HEADERS = {
    "Accept":     "application/sparql-results+json",
    "User-Agent": "kbqa_pipeline/1.0",
}


# ---------------------------------------------------------------------------
# Gold SPARQL normalization

def normalize_gold_sparql(sparql: str, common_prefixes) -> tuple[str | None, str | None]:
    """
    For a given SPARQL query, a normalized version is produced using Apache Jena.
    The query is preprocessed to alleviate common issues that Jena does not accept.
    Afterwards, the query is transformed to Jena Syntax Expression and back to SPARQL.
    """
    try:
        fixed   = fix_sparql_for_jena(sparql, common_prefixes)
        form    = detect_query_form(fixed)
        algebra = sparql_to_algebra(fixed)
        algebra = strip_prefix_and_expand(algebra, common_prefixes)
        result  = algebra_to_sparql(algebra)
        result  = restore_query_form(form, result)
        return result, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# SPARQL execution

def _execute_sparql_raw(sparql: str, endpoint: str, timeout: int):
    """
    Executes a given SPARQL query against the endpoint. No retries.
    """
    resp = requests.post(
        endpoint,
        data={"query": sparql},
        headers=_SPARQL_HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if "boolean" in data:
        return data["boolean"]           # ASK → True / False
    return data.get("results", {}).get("bindings", [])


def execute_sparql(sparql: str, endpoint: str, timeout: int = 30):
    """
    Executes a given SPARQL query against the endpoint. With retries
    and exponential backoff.
    """    
    return call_with_retry(
        _execute_sparql_raw,
        sparql,
        endpoint,
        timeout,
        retries=2,
        base_delay=1.0,
        backoff=2.0,
        exceptions=(requests.RequestException, KeyError, ValueError),
        on_fail=None,
    )


# ---------------------------------------------------------------------------
# Result normalization

def bindings_to_rows(results, kb) -> list[list[str]]:
    """
    Convert raw SPARQL endpoint results into the format required by assignment f1: 
    Each result row becomes a list of KB-local identifiers or literals.

    Handles:
      ASK -> [["true"]] / [["false"]]
      SELECT -> [[val, ...], ...]
    """
    # ASK
    if isinstance(results, bool):
        return [[str(results).lower()]]

    if not isinstance(results, list):
        return []

    rows: list[list[str]] = []

    for row in results:
        if not isinstance(row, dict):
            continue

        values: list[str] = []

        for val in row.values():
            if not isinstance(val, dict):
                continue
            raw = val.get("value", "")
            # Convert full IRI into kb-local identifier
            if val.get("type") == "uri":
                values.append(kb.normalize_answer_uri(raw))
            else:
                lang = val.get("xml:lang", "")
                # Skip non-english literals
                if lang and lang != "en":
                    continue
                stripped = raw.strip()
                if stripped:
                    values.append(stripped.lower())

        if values:
            rows.append(values)

    return rows


def ensure_rows(answer) -> list[list[str]]:
    """
    Force an answer field to list[list[str]] format.
    This function is for legacy support and should not be needed for new runs.
    """
    if not isinstance(answer, list) or not answer:
        return []
    if isinstance(answer[0], str):
        return [[v] for v in answer]
    return [list(row) for row in answer]