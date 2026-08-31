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

_FALLBACK_LOC_RE = re.compile(r"/([^/]+)$")

_normalise_uri: callable = lambda uri: (
    m.group(1) if (m := _FALLBACK_LOC_RE.search(uri)) else uri
)


def init_uri_normaliser(kb_module) -> None:
    global _normalise_uri
    fn = getattr(kb_module, "normalise_answer_uri", None)
    if callable(fn):
        _normalise_uri = fn


# ---------------------------------------------------------------------------
# Gold SPARQL normalisation

def normalise_gold_sparql(sparql: str, common_prefixes) -> tuple[str | None, str | None]:
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
# Result normalisation

def bindings_to_rows(results) -> list[list[str]]:
    """
    Convert raw SPARQL endpoint results into the list-of-rows format required
    by assignment_f1_score: each result row becomes a list of normalised value
    strings, preserving all projected variables.

    Handles:
      ASK  →  [["true"]] / [["false"]]
      SELECT binding dicts  →  [[val, ...], ...]

    Non-English literals are dropped. URI values are reduced to their local
    name via _normalise_uri (KB-specific, set by init_uri_normaliser).
    """
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
            if val.get("type") == "uri":
                values.append(_normalise_uri(raw))
            else:
                lang = val.get("xml:lang", "")
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
    Coerce an answer field to list[list[str]] regardless of how it was stored.

    Handles both the current list-of-rows format and the legacy flat-string
    format produced by older versions of adapt_dataset.py:
        ["m.0135nr", "m.013cqs"]  →  [["m.0135nr"], ["m.013cqs"]]
    """
    if not isinstance(answer, list) or not answer:
        return []
    if isinstance(answer[0], str):
        return [[v] for v in answer]
    return [list(row) for row in answer]