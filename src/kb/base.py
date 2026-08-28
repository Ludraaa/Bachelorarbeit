"""
Abstract base class for KB-agnostic modules.

Concrete KB modules (freebase.py, wikidata.py) subclass BaseKB and
implement/override the KB-specific pieces (label queries, URI patterns,
prefix schemes). Logic that is identical across every KB module we've
written so far -- prefix-table construction/sorting, sexpr entity/relation
extraction via a subclass-supplied regex, and answer-URI normalisation via
an ordered list of subclass-supplied patterns -- lives here so subclasses
only override it if a KB genuinely needs different behaviour.

Anything whose *logic* differs between Freebase and Wikidata (not just its
parameters) is left abstract rather than forced into a shared method, to
avoid silently changing behaviour on either side.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Prefixes shared by every KB module we've written so far. Subclasses
# provide their own KB_PREFIXES (wd/wdt/... or fb/fbp/ns/...) which get
# merged on top of these.

BASE_PREFIXES: dict[str, str] = {
    "rdf":       "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs":      "http://www.w3.org/2000/01/rdf-schema#",
    "owl":       "http://www.w3.org/2002/07/owl#",
    "xsd":       "http://www.w3.org/2001/XMLSchema#",
    "skos":      "http://www.w3.org/2004/02/skos/core#",
    "prov":      "http://www.w3.org/ns/prov#",
    "foaf":      "http://xmlns.com/foaf/0.1/",
    "dct":       "http://purl.org/dc/terms/",
    "dc":        "http://purl.org/dc/elements/1.1/",
    "void":      "http://rdfs.org/ns/void#",
    "qb":        "http://purl.org/linked-data/cube#",
    "schema":    "https://schema.org/",
    "geo":       "http://www.opengis.net/ont/geosparql#",
    "geosparql": "http://www.opengis.net/ont/geosparql#",
    "geof":      "http://www.opengis.net/def/function/geosparql/",
    "gn":        "http://www.geonames.org/ontology#",
    "bd":        "http://www.bigdata.com/rdf#",
    "hint":      "http://www.bigdata.com/queryHints#",
}


class BaseKB(ABC):
    """
    KB-agnostic interface expected by insert_labels.py / resolve_predictions.py.

    Subclasses MUST set as class attributes:
      - KB_PREFIXES:         dict[str, str]    KB-specific prefix -> URI
      - LABEL_QUERY:          str               SPARQL template, {values}/{language}
      - ENTITY_PATTERN:       re.Pattern        matches <...>-wrapped entity URIs
      - RELATION_PATTERN:     re.Pattern        matches <...>-wrapped relation URIs
      - ANSWER_URI_PATTERNS:  list[re.Pattern]  tried in order by normalise_answer_uri;
                               the first match's group(1) is the normalised form

    Subclasses MAY set:
      - LABEL_ENDPOINT_URL:    str | None        SPARQL endpoint this KB's queries must
                               run against (e.g. LABEL_QUERY/TYPES_QUERY rely on a
                               service extension only one endpoint implements).
                               None (the default) means "no opinion" -- insert_labels.py
                               falls back to its own ENDPOINT_URL/$ENDPOINT_URL default
                               in that case. Only override this if the KB's queries are
                               genuinely endpoint-specific, not just "this is what we
                               usually point at".

    Subclasses MUST implement:
      - normalize(uri)
      - parse_label_results(bindings)
      - format_label(uri, label)
      - format_relation_label(uri, label)
      - extract_from_prediction(prediction)
      - substitute(prediction, entity_map, predicate_map, expand_uris)
    """

    LANGUAGE: str = "en"

    KB_PREFIXES: dict[str, str] = {}
    LABEL_QUERY: str = ""
    LABEL_ENDPOINT_URL: str | None = None

    ENTITY_PATTERN: re.Pattern | None = None
    RELATION_PATTERN: re.Pattern | None = None
    ANSWER_URI_PATTERNS: list[re.Pattern] = []

    def __init__(self) -> None:
        self.COMMON_PREFIXES: dict[str, str] = {**BASE_PREFIXES, **self.KB_PREFIXES}
        # Longest base URI first, so the most specific prefix wins if two
        # prefixes' URIs happen to be prefixes of one another.
        self._prefix_lookup: list[tuple[str, str]] = sorted(
            self.COMMON_PREFIXES.items(), key=lambda kv: len(kv[1]), reverse=True
        )

    # -- shared: sexpr URI extraction ---------------------------------------

    def extract_entities(self, sexpr: str) -> list[str]:
        """Return all entity URIs present in *sexpr* (wrapped in <...>)."""
        return list(set(self.ENTITY_PATTERN.findall(sexpr)))

    def extract_relations(self, sexpr: str) -> list[str]:
        """Return all relation/property URIs present in *sexpr* (wrapped in <...>)."""
        return list(set(self.RELATION_PATTERN.findall(sexpr)))

    # -- shared: answer URI normalisation ------------------------------------

    def normalise_answer_uri(self, uri: str) -> str:
        """
        Strip a full URI down to its KB-local identifier by trying each of
        ANSWER_URI_PATTERNS in order and returning the first match's group(1).
        Falls back to whatever follows the last "/" if nothing matches.
        """
        for pattern in self.ANSWER_URI_PATTERNS:
            m = pattern.match(uri)
            if m:
                return m.group(1)
        return uri.rsplit("/", 1)[-1]

    # -- shared helper available to format_label() implementations -----------

    def _format_via_common_prefixes(self, uri: str) -> str:
        """
        Fallback used once KB-specific entity/relation checks in
        format_label() have failed: map uri -> "prefix:local" using
        COMMON_PREFIXES, falling back to the bare local name. This is the
        exact logic freebase.py's generic branch used. wikidata.py additionally
        prefers a fetched label over the local name in this fallback, so it
        does not reuse this helper as-is -- see wikidata.py's format_label.
        """
        for prefix, base in self._prefix_lookup:
            if uri.startswith(base):
                local = uri[len(base):]
                return f"{prefix}:{local}" if local else ""
        return ""

    # -- KB-specific: must be implemented by subclasses -----------------------

    @abstractmethod
    def normalize(self, uri: str) -> str | None:
        """Map uri to the canonical URI carrying its label-bearing triple, or None."""

    @abstractmethod
    def parse_label_results(self, bindings: list[dict]) -> dict[str, str]:
        """Parse LABEL_QUERY SPARQL bindings into {uri: label}."""

    @abstractmethod
    def format_label(self, uri: str, label: str) -> str:
        """Format uri (with its fetched label, if any) as a prefixed sexpr token."""

    @abstractmethod
    def format_relation_label(self, uri: str, label: str) -> str | None:
        """Format uri for use as a gold_relation_map value."""

    @abstractmethod
    def extract_from_prediction(self, prediction: str) -> tuple[list[str], list[str]]:
        """Extract (entity_labels, predicate_labels) tokens from a raw model prediction."""

    @abstractmethod
    def substitute(
        self,
        prediction: str,
        entity_map: dict[str, str],
        predicate_map: dict[str, str],
        expand_uris: bool = True,
    ) -> str:
        """Replace linked entity/predicate tokens in prediction with resolved URIs."""