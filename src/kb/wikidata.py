import re

COMMON_PREFIXES: dict[str, str] = {
    "xsd":      "http://www.w3.org/2001/XMLSchema#",
    "wd":       "http://www.wikidata.org/entity/",
    "wdt":      "http://www.wikidata.org/prop/direct/",
    "p":        "http://www.wikidata.org/prop/",
    "ps":       "http://www.wikidata.org/prop/statement/",
    "pq":       "http://www.wikidata.org/prop/qualifier/",
    "psv":      "http://www.wikidata.org/prop/statement/value/",
    "psn":      "http://www.wikidata.org/prop/statement/value-normalized/",
    "pqv":      "http://www.wikidata.org/prop/qualifier/value/",
    "pqn":      "http://www.wikidata.org/prop/qualifier/value-normalized/",
    "pr":       "http://www.wikidata.org/prop/reference/",
    "prv":      "http://www.wikidata.org/prop/reference/value/",
    "prn":      "http://www.wikidata.org/prop/reference/value-normalized/",
    "wikibase": "http://wikiba.se/ontology#",
    "schema":   "https://schema.org/",
    "rdf":      "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs":     "http://www.w3.org/2000/01/rdf-schema#",
    "owl":      "http://www.w3.org/2002/07/owl#",
    "xsd":      "http://www.w3.org/2001/XMLSchema#",
    "foaf":     "http://xmlns.com/foaf/0.1/",
    "skos":     "http://www.w3.org/2004/02/skos/core#",
    "dct":      "http://purl.org/dc/terms/",
    "dc":       "http://purl.org/dc/elements/1.1/",
    "qb":       "http://purl.org/linked-data/cube#",
    "prov":     "http://www.w3.org/ns/prov#",
    "geo":      "http://www.opengis.net/ont/geosparql#",
    "geosparql":"http://www.opengis.net/ont/geosparql#",
    "geof":     "http://www.opengis.net/def/function/geosparql/",
    "gn":       "http://www.geonames.org/ontology#",
    "bd":       "http://www.bigdata.com/rdf#",
    "hint":     "http://www.bigdata.com/queryHints#",
    "void":     "http://rdfs.org/ns/void#",
}

LANGUAGE = "en"

# SPARQL template: insert_labels.py substitutes {values} and {language}.
LABEL_QUERY = """
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX bd:       <http://www.bigdata.com/rdf#>
    PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?uri ?uriLabel ?directLabel WHERE {{
        VALUES ?uri {{ {values} }}
        SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "{language},en".
        }}
        OPTIONAL {{
            ?uri rdfs:label ?directLabel .
            FILTER(LANG(?directLabel) = "en")
        }}
    }}
"""

# ---------------------------------------------------------------------------
# URI patterns

_PID_PATTERN = re.compile(
    r"http://www\.wikidata\.org/prop(?:/direct|/statement|/qualifier|/reference)?/(P\d+)"
)
_QP_PATTERN  = re.compile(r"http://www\.wikidata\.org/entity/([QP]\d+)")

_Q_IN_SEXPR  = re.compile(r"<(http://www\.wikidata\.org/entity/Q\d+)>")
_P_IN_SEXPR  = re.compile(
    r"<(http://www\.wikidata\.org/(?:entity/P\d+|prop(?:/[^/]+)?/P\d+))>"
)

def normalise_answer_uri(uri: str) -> str:
    """Strip a Wikidata URI to its QID or PID."""
    m = _QP_PATTERN.match(uri)
    if m:
        return m.group(1)          # "Q2720627" or "P31"
    m = _PID_PATTERN.match(uri)
    if m:
        return m.group(1)          # "P31" from any prop/ variant
    return uri.rsplit("/", 1)[-1]  # fallback


# ---------------------------------------------------------------------------
# normalize  (REQUIRED by insert_labels.py)

def normalize(uri: str) -> str | None:
    """
    Map any Wikidata URI to the canonical entity URI that carries its label.
    Property URIs (prop/direct/P31, etc.) are mapped to entity/P31 because
    rdfs:label lives on the entity URI.
    Returns None for unrecognised URIs (they are skipped silently).
    """
    pid_match = _PID_PATTERN.match(uri)
    if pid_match:
        return f"http://www.wikidata.org/entity/{pid_match.group(1)}"
    if _QP_PATTERN.match(uri):
        return uri     # already canonical
    return None


# ---------------------------------------------------------------------------
# parse_label_results  (REQUIRED by insert_labels.py)

def parse_label_results(bindings: list[dict]) -> dict[str, str]:
    """
    Parse results from LABEL_QUERY.
    Prefers the wikibase:label service value (uriLabel) and falls back to a
    direct rdfs:label. Skips rows where the label is just the bare entity ID.
    """
    result: dict[str, str] = {}
    for row in bindings:
        uri        = row.get("uri",         {}).get("value", "")
        uri_label  = row.get("uriLabel",    {}).get("value", "")
        direct     = row.get("directLabel", {}).get("value", "")
        if not uri:
            continue
        entity_id = uri.rsplit("/", 1)[-1]
        label = (uri_label if uri_label and uri_label != entity_id else None) or direct or None
        if label:
            result[uri] = label
    return result


# ---------------------------------------------------------------------------
# format_label  (controls sexpr_with_labels)

# Build once at import time — longest base URI first guarantees most-specific match
_PREFIX_LOOKUP: list[tuple[str, str]] = sorted(
    COMMON_PREFIXES.items(), key=lambda x: len(x[1]), reverse=True
)

def format_label(uri: str, label: str) -> str:
    # case: entity
    if "/entity/Q" in uri:
        return f"wd:{label.replace(' ', '_')}" if label else ""

    # case: property
    is_property = "/entity/P" in uri or "/prop/" in uri
    slug = label.replace(" ", "_").lower() if (label and is_property) else (label or "")

    # case: COMMON_PREFIXES
    for prefix, base in _PREFIX_LOOKUP:
        if uri.startswith(base):
            # fall back to the local name extracted from the URI itself
            local = uri[len(base):]          # "label" from rdfs:label
            effective_slug = slug or local   # prefer fetched label, else local name
            return f"{prefix}:{effective_slug}" if effective_slug else ""

    return ""


# ---------------------------------------------------------------------------
# format_relation_label  (controls gold_relation_map values)

def format_relation_label(uri: str, label: str) -> str:
    return label


# ---------------------------------------------------------------------------
# URI extraction from sexprs

def extract_entities(sexpr: str) -> list[str]:
    """Return all Q-item URIs present in *sexpr*."""
    return list(set(_Q_IN_SEXPR.findall(sexpr)))


def extract_relations(sexpr: str) -> list[str]:
    """Return all property URIs present in *sexpr* (any prop/ variant)."""
    return list(set(_P_IN_SEXPR.findall(sexpr)))


# ---------------------------------------------------------------------------
# Prediction post-processing  (used by resolve_predictions.py)

import re

_PRED_ENTITY_RE    = re.compile(r"\bwd:(\S+)")
_PRED_PREDICATE_RE = re.compile(r"\bwdt:(\S+)")

def extract_from_prediction(prediction: str):
    entities   = [e.rstrip(");.") for e in _PRED_ENTITY_RE.findall(prediction)]
    predicates = [p.rstrip(");.") for p in _PRED_PREDICATE_RE.findall(prediction)]

    return list(dict.fromkeys(entities)), list(dict.fromkeys(predicates))


def substitute(prediction: str,
               entity_map: dict[str, str],
               predicate_map: dict[str, str],
               expand_uris: bool = True) -> str:
    """
    Replace wd:Label → QID and wdt:Label → PID, then optionally expand all
    remaining prefix:local tokens to full URIs using COMMON_PREFIXES.
    """
    for label, qid in entity_map.items():
        replacement = f"<{COMMON_PREFIXES['wd']}{qid}>" if expand_uris else f"wd:{qid}"
        prediction  = prediction.replace(f"wd:{label}", replacement)

    for label, pid in predicate_map.items():
        replacement = f"<{COMMON_PREFIXES['wdt']}{pid}>" if expand_uris else f"wdt:{pid}"
        prediction  = prediction.replace(f"wdt:{label}", replacement)

    if expand_uris:
        for prefix, base_uri in COMMON_PREFIXES.items():
            if prefix in ("wd", "wdt"):
                continue
            pattern    = re.compile(rf"(?<!<)\b{re.escape(prefix)}:([A-Za-z0-9_\-\.]+)")
            prediction = pattern.sub(lambda m, b=base_uri: f"<{b}{m.group(1)}>", prediction)

    return prediction