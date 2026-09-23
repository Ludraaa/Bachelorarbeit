"""
Abstract base class for KB-agnostic modules.
Concrete KB modules (like freebase.py, wikidata.py) extend BaseKB and implement the KB-specific pieces defined in this file.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Common prefixes shared across knowledge bases.
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
      - LABEL_QUERY: str               
        
        Defines a template SPARQL query that is used to query the endpoint for labels.
        These labels are substituted into the generated training format and forms the basis
        of model finetuning as well as entity and predicate retrieval.
        The label insertion step of the pipeline uses the label query in the following way:
        
            kb.LABEL_QUERY.format(values=values, language=kb.LANGUAGE)
            
        As such, the label query MUST accept a {values} variable, and CAN accept a {language}
        variable if the concrete query can make use of it.
      
      - ENTITY_PATTERN: re.Pattern
           
        Defines a REGEX to extract full <...>-wrapped entity IRIs from the training format.
        Only has to handle IRIs that are specific to the knowledge base. 
        This is used by the label insertion step to replace full entity IRIs with prefixed labels.
        
      - RELATION_PATTERN: re.Pattern        
      
        Defines a REGEX to extract full <...>-wrapped predicate IRIs from the training format.
        This has a similar purpose as ENTITY_PATTERN.
        
      - ANSWER_URI_PATTERNS: list[re.Pattern]  
      
        A list of patterns that is sequentially tried in normalize_answer_uri().
        This is used to normalize a full IRI query result to a local identifier.
        If all fail, the last '/' is used as a fallback.


    Subclasses MAY set:
      - KB_PREFIXES: dict[str, str]
        Default: {}
          
        Additional prefix map needed for prefixes unique to the target knowledge base.
        
      - LABEL_ENDPOINT_URL: str | None
      
        Defines a separate endpoint for label queries only. This can be used to execute label queries
        against a public endpoint that defines various extensions helpful for fetching labels, 
        while the later retrieval step can not be used on a public endpoint and instead has to point
        the environment var 'ENDPOINT_URL' to a private instance.
        
      - TYPES_QUERY: str | None
      
        Optionally defines a SPARQL query that determines whether a certain entity is a class/type
        in the context of the knowledge base. These are collected into a separate train_type_map by
        the original ChatKBQA approach. During retrieval, these can be used by linker implementations.
        
        Similarly to LABEL_QUERY, it MUST accept a {values} variable.


    Subclasses MUST implement:
      - normalize(uri)
      
        For any given input IRI, returns the actual label-carrying IRI.
        Take a look at the Freebase and Wikidata versions of this function
        for a better idea of what this function can be used for.
        
      - parse_label_results(bindings)
      
        Takes the bindings of the LABEL_QUERY and is responsible for choosing
        exactly one of the results as the final label.
        
      - format_label(uri, label)
      
        Given an IRI and a (not strictly required; see Freebase) label, controls what the
        prefixed version of it used in label-enriched training data looks like.
        Probably includes using _format_via_common_prefixes() to handle common human-readable
        identifiers like 'rdfs:label'.
        
      - extract_from_prediction(prediction)
      
        Given a prediction beam, this method returns a tuple containing two lists. The first list
        contains all detected entity mentions, while the second contains all detected predicate
        mentions. This method is used to determine mentions that need to be resolved from 
        label -> local KB identifier during the resolution/retrieval step.
        
      - substitute(prediction, entity_map, predicate_map)
      
        Given a prediction beam, entity and predicate map, this method is responsible for inserting
        the resolved local kb identifiers back into the prediction beam. This method should also expand
        any prefixed identifiers to full IRIs.
    """

    LANGUAGE: str = "en"

    KB_PREFIXES: dict[str, str] = {}
    LABEL_QUERY: str
    LABEL_ENDPOINT_URL: str | None = None
    TYPES_QUERY: str | None = None

    ENTITY_PATTERN: re.Pattern
    RELATION_PATTERN: re.Pattern
    ANSWER_URI_PATTERNS: list[re.Pattern] = []


    def __init__(self) -> None:
        required_attributes = (
            "LABEL_QUERY",
            "ENTITY_PATTERN",
            "RELATION_PATTERN",
        )

        missing = [
            name
            for name in required_attributes
            if getattr(self, name, None) is None
        ]

        if missing:
            raise TypeError(
                f"{type(self).__name__} is missing required KB attributes: "
                + ", ".join(missing)
            )

        self.COMMON_PREFIXES: dict[str, str] = {
            **BASE_PREFIXES,
            **self.KB_PREFIXES,
        }

        # Longest base URI first, so the most specific prefix wins if
        # prefix IRIs happen to be prefixes of each other.
        self._prefix_lookup: list[tuple[str, str]] = sorted(
            self.COMMON_PREFIXES.items(),
            key=lambda kv: len(kv[1]),
            reverse=True,
        )

    # ---------------------------------------------------------------------------
    # Shared behaviour

    # Extraction from training format

    def extract_entities(self, sexpr: str) -> list[str]:
        """
        Returns all entity IRIs present in training format.
        Controlled by ENTITY_PATTERN.
        """
        return list(set(self.ENTITY_PATTERN.findall(sexpr)))

    def extract_relations(self, sexpr: str) -> list[str]:
        """
        Returns all predicate IRIs present in training format.
        Controlled by RELATION_PATTERN.
        """
        return list(set(self.RELATION_PATTERN.findall(sexpr)))


    # Answer normalization

    def normalize_answer_uri(self, uri: str) -> str:
        """
        Strip a full IRI down to a local identifier. 
        Tries all defined ANSWER_URI_PATTERNS in order and returns the 
        first match's group(1).
        Falls back to whatever follows the last '/' if nothing matches.
        """
        for pattern in self.ANSWER_URI_PATTERNS:
            m = pattern.match(uri)
            if m:
                return m.group(1)
        return uri.rsplit("/", 1)[-1]


    # Type-URI filtering

    def parse_type_results(self, bindings: list[dict]) -> set[str]:
        """
        Returns the set of IRIs from bindings that act as types/classes in this KB.
        """
        return {row["uri"]["value"] for row in bindings if "uri" in row}


    # Possible fallback for format_labels()

    def _format_via_common_prefixes(self, uri: str) -> str:
        """
        Falls back to prefix:bare_local_name, based on the first matching common prefix. 
        This is used intentionally by Freebase, as predicates do not have
        labels in freebase, and are instead derived from the IRI itself.
        """
        for prefix, base in self._prefix_lookup:
            if uri.startswith(base):
                local = uri[len(base):]
                return f"{prefix}:{local}" if local else ""
        return ""

    def format_relation_label(self, uri: str, label: str) -> str | None:
        """Format uri for use as a gold_relation_map value."""
        return label

    # ---------------------------------------------------------------------------
    # KB-specific behaviour

    # Label insertion

    @abstractmethod
    def normalize(self, uri: str) -> str | None:
        """Maps a given IRI to the IRI carrying its label-bearing triple, or return None."""

    @abstractmethod
    def parse_label_results(self, bindings: list[dict]) -> dict[str, str]:
        """Parses LABEL_QUERY SPARQL bindings into {uri: label}."""

    @abstractmethod
    def format_label(self, uri: str, label: str) -> str:
        """Formats uri (with its fetched label, if any) as a prefixed training format token."""


    # Resolve / Retrieve

    @abstractmethod
    def extract_from_prediction(self, prediction: str) -> tuple[list[str], list[str]]:
        """Extracts (entity_labels, predicate_labels) tokens from a raw model prediction."""

    @abstractmethod
    def substitute(
        self,
        prediction: str,
        entity_map: dict[str, str],
        predicate_map: dict[str, str],
        expand_uris: bool = True,
    ) -> str:
        """Replaces linked entity/predicate tokens in prediction with resolved IRIs."""