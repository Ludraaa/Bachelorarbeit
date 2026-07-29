import re

# ---------------------------------------------------------------------------
# Common prefixes

COMMON_PREFIXES: dict[str, str] = {
    # Freebase-specific
    "fb":       "http://rdf.freebase.com/ns/",
    "fbp":      "http://rdf.freebase.com/ns/",
    "ns":       "http://rdf.freebase.com/ns/",

    "rdf":      "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs":     "http://www.w3.org/2000/01/rdf-schema#",
    "owl":      "http://www.w3.org/2002/07/owl#",
    "xsd":      "http://www.w3.org/2001/XMLSchema#",
    "skos":     "http://www.w3.org/2004/02/skos/core#",
    "prov":     "http://www.w3.org/ns/prov#",

    "foaf":     "http://xmlns.com/foaf/0.1/",
    "dct":      "http://purl.org/dc/terms/",
    "dc":       "http://purl.org/dc/elements/1.1/",
    "void":     "http://rdfs.org/ns/void#",
    "qb":       "http://purl.org/linked-data/cube#",
    "schema":   "https://schema.org/",

    "geo":      "http://www.opengis.net/ont/geosparql#",
    "geosparql":"http://www.opengis.net/ont/geosparql#",
    "geof":     "http://www.opengis.net/def/function/geosparql/",
    "gn":       "http://www.geonames.org/ontology#",
    "bd":       "http://www.bigdata.com/rdf#",
    "hint":     "http://www.bigdata.com/queryHints#",
}

LANGUAGE = "en"

_FB_NS = "http://rdf.freebase.com/ns/"

# ---------------------------------------------------------------------------
# SPARQL label query
#
# Only entity MIDs (m.xxx / g.xxx) carry type.object.name triples in the
# standard Freebase RDF dump.  Relation URIs have no label triple at all;
# normalize() returns None for them so insert_labels.py skips the batch
# lookup and format_label() derives their display token from the URI path.

LABEL_QUERY = """
    SELECT ?uri ?label WHERE {{
        VALUES ?uri {{ {values} }}
        ?uri <http://rdf.freebase.com/ns/type.object.name> ?label .
        FILTER(LANG(?label) = "{language}" || LANG(?label) = "")
    }}
"""

# ---------------------------------------------------------------------------
# SPARQL types query
#
# Determines which entity URIs are Freebase *type* entities.

TYPES_QUERY = """
SELECT ?uri WHERE {{
    VALUES ?uri {{ {values} }}
    ?uri <http://rdf.freebase.com/ns/type.object.type>
         <http://rdf.freebase.com/ns/type.type> .
}}
"""


def parse_type_results(bindings: list[dict]) -> set[str]:
    """Return the set of URIs from *bindings* that are Freebase types."""
    return {
        row["uri"]["value"]
        for row in bindings
        if "uri" in row
    }

# ---------------------------------------------------------------------------
# URI patterns (internal)
#
# MID  : m.0f8l9c  /  g.119pgc8
# Relation: domain.type.property

_MID_RE = re.compile(
    r"^http://rdf\.freebase\.com/ns/([mg]\.[0-9a-z_]+)$"
)
_REL_RE = re.compile(
    r"^http://rdf\.freebase\.com/ns/"
    r"(?!(?:m|g)\.)"                       # exclude MIDs
    r"([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){2,})$"  # ≥3 dot-segments
)

# For extracting URIs from sexprs (full URIs wrapped in angle brackets)
_Q_IN_SEXPR = re.compile(
    r"<(http://rdf\.freebase\.com/ns/[mg]\.[0-9a-z_]+)>"
)
_P_IN_SEXPR = re.compile(
    r"<(http://rdf\.freebase\.com/ns/"
    r"(?!(?:m|g)\.)"
    r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){2,})>"
)


def normalise_answer_uri(uri: str) -> str:
    """Strip a Freebase URI to its local MID or relation path."""
    m = _MID_RE.match(uri)
    if m:
        return m.group(1)          # "m.01428y"
    m = _REL_RE.match(uri)
    if m:
        return m.group(1)          # "people.person.date_of_birth"
    return uri.rsplit("/", 1)[-1]  # fallback: anything after last /


# ---------------------------------------------------------------------------
# normalize  (REQUIRED by insert_labels.py)

def normalize(uri: str) -> str | None:
    """
    Entity MIDs already carry their label (type.object.name), so they are
    returned as-is.

    Relation URIs have no label triple in Freebase; returning None tells
    insert_labels.fetch_labels() to write cache[uri] = None immediately and
    skip the SPARQL batch.  format_label() will then derive the token from
    the URI local name.
    """
    if _MID_RE.match(uri):
        return uri
    return None


# ---------------------------------------------------------------------------
# parse_label_results  (REQUIRED by insert_labels.py)

def parse_label_results(bindings: list[dict]) -> dict[str, str]:
    preferred: dict[str, str] = {}   # uri -> en label
    fallback:  dict[str, str] = {}   # uri -> first non-en label

    for row in bindings:
        uri   = row.get("uri",   {}).get("value", "")
        label = row.get("label", {}).get("value", "")
        lang  = row.get("label", {}).get("xml:lang", "")
        if not uri or not label:
            continue
        if lang == "en":
            preferred[uri] = label
        elif uri not in fallback:
            fallback[uri] = label

    result = {**fallback, **preferred}   # preferred overwrites fallback
    return result


# ---------------------------------------------------------------------------
# format_label  (controls sexpr_with_labels output)

# Build once at import time
_PREFIX_LOOKUP: list[tuple[str, str]] = sorted(
    COMMON_PREFIXES.items(), key=lambda x: len(x[1]), reverse=True
)

def format_label(uri: str, label: str) -> str:
    """
    Entity MID  → fb:Human_Readable_Label
                  (falls back to fb:m.0f8l9c if no name was found)

    Relation    → fbp:domain.type.property
    """
    if _MID_RE.match(uri):
        local = uri[len(_FB_NS):]                  # e.g. "m.0f8l9c"
        slug  = label.replace(" ", "_") if label else local
        return f"fb:{slug}"

    if _REL_RE.match(uri):
        local = uri[len(_FB_NS):]                  # e.g. "people.person.date_of_birth"
        return f"fbp:{local}"

    # case: COMMON_PREFIXES
    for prefix, base in _PREFIX_LOOKUP:
        if uri.startswith(base):
            local = uri[len(base):]
            return f"{prefix}:{local}" if local else ""


    return ""


# ---------------------------------------------------------------------------
# format_relation_label  (controls gold_relation_map values)

def format_relation_label(uri: str, label: str) -> str:
    if _REL_RE.match(uri):
        return uri[len(_FB_NS):]
    return None

# ---------------------------------------------------------------------------
# URI extraction from sexprs  (REQUIRED by insert_labels.py)

def extract_entities(sexpr: str) -> list[str]:
    """Return all entity MID URIs present in *sexpr*."""
    return list(set(_Q_IN_SEXPR.findall(sexpr)))


def extract_relations(sexpr: str) -> list[str]:
    """Return all relation URIs present in *sexpr*."""
    return list(set(_P_IN_SEXPR.findall(sexpr)))


# ---------------------------------------------------------------------------
# Prediction post-processing  (used by resolve_predictions.py)

import re

_PREFIX_RE = re.compile(r"\bfb(p)?:\s*")


def _scan_token(s: str, start: int) -> str:
    depth = 0
    i = start
    n = len(s)
    while i < n:
        c = s[i]
        if c in " \t\n\r" or c == ";":
            break
        if c == "(":
            depth += 1
            i += 1
            continue
        if c == ")":
            if depth > 0:
                depth -= 1
                i += 1
                continue
            break  # unbalanced close
        i += 1
    return s[start:i]


def extract_from_prediction(prediction: str):
    entities, predicates = [], []
    for m in _PREFIX_RE.finditer(prediction):
        is_pred = m.group(1) is not None
        token = _scan_token(prediction, m.end()).rstrip(",")
        if not token:
            continue
        (predicates if is_pred else entities).append(token)

    return list(dict.fromkeys(entities)), list(dict.fromkeys(predicates))

def substitute(
    prediction: str,
    entity_map: dict[str, str],
    predicate_map: dict[str, str],
    expand_uris: bool = True,
) -> str:
    """
    Replace fb:Label → MID and fbp:path → relation path, then optionally
    expand all remaining prefix:local tokens to full URIs.
    """
    for label, mid in entity_map.items():
        replacement = f"<http://rdf.freebase.com/ns/{mid}>"
        prediction = prediction.replace(f"fb:{label}", replacement)

    for label, rel in predicate_map.items():
        replacement = f"<http://rdf.freebase.com/ns/{rel}>"
        prediction = prediction.replace(f"fbp:{label}", replacement)

    if expand_uris:
        # Expand any remaining prefix:local tokens (rdf:, rdfs:, xsd:, …)
        # Skip fb/fbp: was unlinked and should stay as-is
        for prefix, base_uri in COMMON_PREFIXES.items():
            if prefix in ("fb", "fbp"):
                continue
            pattern    = re.compile(rf"(?<!<)\b{re.escape(prefix)}:([A-Za-z0-9_\-\.]+)")
            prediction = pattern.sub(lambda m, b=base_uri: f"<{b}{m.group(1)}>", prediction)

    return prediction

