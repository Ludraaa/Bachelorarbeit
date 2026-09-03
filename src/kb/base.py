"""
Abstract base class for KB-agnostic modules.

Concrete KB modules (like freebase.py, wikidata.py) extend BaseKB and implement the KB-specific pieces defined in this file.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Shared prefixes. Subclasses additionally provide their own KB_PREFIXES (wd/wdt/... or fb/fbp/ns/...) which get merged on top of these.

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
    KB-agnostic interface.

    Subclasses MUST set as class attributes:
      - KB_PREFIXES:         dict[str, str]    KB-specific prefix -> URI
      - LABEL_QUERY:         str               SPARQL template, {values}/{language}
      - ENTITY_PATTERN:      re.Pattern        matches <...>-wrapped entity URIs
      - RELATION_PATTERN:    re.Pattern        matches <...>-wrapped predicate URIs
      - ANSWER_URI_PATTERNS: list[re.Pattern]  tried in order by normalise_answer_uri;
                             the first match's group(1) is the normalised form

    Subclasses MAY set:
      - LABEL_ENDPOINT_URL: str | None        SPARQL endpoint this KB's queries must
                            run against (if, for example, LABEL_QUERY/TYPES_QUERY rely on a
                            service extension only one endpoint implements).
                            Runs on the endpoint specified in the run config if not explicitly set.

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
        # prefix URIs happen to be prefixes of each another.
        self._prefix_lookup: list[tuple[str, str]] = sorted(
            self.COMMON_PREFIXES.items(), key=lambda kv: len(kv[1]), reverse=True
        )

    # ---------------------------------------------------------------------------
    # Shared behaviour

    # extraction from sexpr

    def extract_entities(self, sexpr: str) -> list[str]:
        """Return all entity URIs present in sexpr (wrapped in <...>)."""
        return list(set(self.ENTITY_PATTERN.findall(sexpr)))

    def extract_relations(self, sexpr: str) -> list[str]:
        """Return all predicate URIs present in sexpr (wrapped in <...>)."""
        return list(set(self.RELATION_PATTERN.findall(sexpr)))


    # answer normalization

    def normalise_answer_uri(self, uri: str) -> str:
        """
        Strip a full URI down to the KB-local identifier. Tries all defined
        ANSWER_URI_PATTERNS in order and returns the first match's group(1).
        Falls back to whatever follows the last "/" if nothing matches.
        """
        for pattern in self.ANSWER_URI_PATTERNS:
            m = pattern.match(uri)
            if m:
                return m.group(1)
        return uri.rsplit("/", 1)[-1]


    # fallback for format_labels()

    def _format_via_common_prefixes(self, uri: str) -> str:
        """
        Fallback used once KB-specific entity/relation checks in
        format_label() have failed: map uri -> "prefix:local" using
        COMMON_PREFIXES, falling back to the bare local name. This is
        used intentionally by freebase.py, as predicates do not have
        labels in freebase, and are instead derived from the uri itself.
        """
        for prefix, base in self._prefix_lookup:
            if uri.startswith(base):
                local = uri[len(base):]
                return f"{prefix}:{local}" if local else ""
        return ""

    # ---------------------------------------------------------------------------
    # KB-specific behaviour

    # label insertion

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


    # resolve

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