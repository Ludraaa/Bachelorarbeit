import re

from src.kb.base import BaseKB


_WD_NS = "http://www.wikidata.org/entity/"
_WDT_NS = "http://www.wikidata.org/prop/direct/"

_PID_PATTERN = re.compile(
    r"http://www\.wikidata\.org/prop(?:/direct|/statement|/qualifier|/reference)?/(P\d+)"
)
_QP_PATTERN = re.compile(r"http://www\.wikidata\.org/entity/([QP]\d+)")

_Q_IN_SEXPR = re.compile(r"<(http://www\.wikidata\.org/entity/Q\d+)>")
_P_IN_SEXPR = re.compile(
    r"<(http://www\.wikidata\.org/(?:entity/P\d+|prop(?:/[^/]+)?/P\d+))>"
)

PREDICATE_PREFIXES: tuple[str, ...] = tuple(
    sorted(("wdt", "p", "ps", "pq", "psv", "psn", "pqv", "pqn", "pr", "prv", "prn"),
           key=len, reverse=True)
)

_PATH_PREFIX_LOOKAHEAD = re.compile(
    r"^\^?(?:" + "|".join(PREDICATE_PREFIXES) + r"):"
)


def _looks_like_path_continuation(text: str, pos: int) -> bool:
    """
    True if text[pos:] is a SPARQL path symbol, not label content.
    Path composition always re-prefixes each atom, so a prefix right
    after the symbol means syntax; anything else means label text.
    """
    rest = text[pos:]
    if rest.startswith("("):
        rest = rest[1:]
    return bool(_PATH_PREFIX_LOOKAHEAD.match(rest))


def _consume_predicate_label(text: str, start: int) -> str:
    depth = 0  # parens opened as LABEL content, not path groups
    i = start
    n = len(text)
    while i < n:
        ch = text[i]

        if ch in " \t\n":
            break

        if ch in "*+^":
            break  # quantifiers/inverse

        if ch in "/|":
            if _looks_like_path_continuation(text, i + 1):
                break  # genuine path separator
            i += 1
            continue  # label content, like "located_in/on_physical_feature"

        if ch == "(":
            if depth == 0 and _looks_like_path_continuation(text, i + 1):
                break  # genuine path-group opener, like "/(wdt:subclass_of)*"
            depth += 1
            i += 1
            continue  # label content, like "has_part(s)"

        if ch == ")":
            if depth == 0:
                break  # closes an enclosing structural group, not ours
            depth -= 1
            i += 1
            continue

        if ch in ".,;":
            # A label's own trailing punctuation and a SPARQL terminator
            # glued on with no space look identical from the string alone.
            # Treat as structural only when directly followed by whitespace
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == "" or nxt in " \t\n":
                break
            i += 1
            continue

        i += 1

    return text[start:i]


def _consume_entity_label(text: str, start: int) -> str:
    # Entities never sit in path-expression position, so no operator
    # lookahead is needed. '/', '*', '+' etc. always pass through as
    # label content.
    depth = 0
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\n":
            break
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                break
            depth -= 1
        elif ch in ".,;":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == "" or nxt in " \t\n":
                break
        i += 1
    return text[start:i]


_ENTITY_ANCHOR_RE = re.compile(r"\bwd:")
_PREDICATE_ANCHOR_RE = re.compile(
    r"\b(?:" + "|".join(PREDICATE_PREFIXES) + r"):"
)


class Wikidata(BaseKB):

    KB_PREFIXES = {
        "wd":       _WD_NS,
        "wdt":      _WDT_NS,
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
    }

    #   0. {language}
    #   1. "mul" -- language-neutral fallback label
    #   2. anything else
    LABEL_QUERY = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?uri ?label ?lang WHERE {{
            VALUES ?uri {{ {values} }}
            ?uri rdfs:label ?label .
            BIND(LANG(?label) AS ?lang)
            BIND(
                IF(?lang = "{language}", 0,
                    IF(?lang = "mul", 1, 2)
                ) AS ?priority
            )
        }}
        ORDER BY ?uri ?priority
    """

    # Wikidata has no specified types. Define an entity as a type, if either:
    # 1. Another entity is P31 ("instance of") the entity
    # 2. Another entity is P279 ("subclass of") the entity
    # This is not perfect (even a single use as instance or subclass of -> is type), but good enough
    TYPES_QUERY = """
        SELECT DISTINCT ?uri WHERE {{
            VALUES ?uri {{ {values} }}

            {{
                ?instance <http://www.wikidata.org/prop/direct/P31> ?uri .
            }}
            UNION
            {{
                ?subclass <http://www.wikidata.org/prop/direct/P279> ?uri .
            }}
        }}
    """

    ENTITY_PATTERN = _Q_IN_SEXPR
    RELATION_PATTERN = _P_IN_SEXPR
    ANSWER_URI_PATTERNS = [_QP_PATTERN, _PID_PATTERN]
    PREDICATE_PREFIXES = PREDICATE_PREFIXES

    # ---------------------------------------------------------------------------
    # label insertion

    def normalize(self, uri: str) -> str | None:
        """
        Entities carry their own label, but a predicate label lives on the
        predicate's entity instead.
        """
        pid_match = _PID_PATTERN.match(uri)
        if pid_match:
            return f"{_WD_NS}{pid_match.group(1)}"
        if _QP_PATTERN.match(uri):
            return uri  # already canonical
        return None


    def parse_label_results(self, bindings: list[dict]) -> dict[str, str]:
        """
        Takes the first label per URI based on the SPARQL ordering in the query.
        """
        result: dict[str, str] = {}
        
        for row in bindings:
            uri = row.get("uri", {}).get("value", "")
            label = row.get("label", {}).get("value", "")
            
            if not uri or not label:
                continue
            
            # Only take the first label per URI
            if uri not in result:
                entity_id = uri.rsplit("/", 1)[-1]
                # Skip if label is just the entity ID
                if label != entity_id:
                    result[uri] = label
        
        return result

    def format_label(self, uri: str, label: str) -> str:
        # case: entity
        if "/entity/Q" in uri:
            return f"wd:{label.replace(' ', '_')}" if label else ""

        # case: property
        is_property = "/entity/P" in uri or "/prop/" in uri
        slug = label.replace(" ", "_").lower() if (label and is_property) else (label or "")

        # case: COMMON_PREFIXES - prefer the fetched label
        for prefix, base in self._prefix_lookup:
            if uri.startswith(base):
                local = uri[len(base):]
                effective_slug = slug or local
                return f"{prefix}:{effective_slug}" if effective_slug else ""

        return ""


    def format_relation_label(self, uri: str, label: str) -> str | None:
        return label


    # ---------------------------------------------------------------------------
    # prediction extraction and substitution (resolve step)

    def extract_from_prediction(self, prediction: str) -> tuple[list[str], list[str]]:
        entities = []
        for m in _ENTITY_ANCHOR_RE.finditer(prediction):
            label = _consume_entity_label(prediction, m.end())
            if label:
                entities.append(label)

        predicates = []
        for m in _PREDICATE_ANCHOR_RE.finditer(prediction):
            label = _consume_predicate_label(prediction, m.end())
            if label:
                predicates.append(label)

        return list(dict.fromkeys(entities)), list(dict.fromkeys(predicates))


    def substitute(
        self,
        prediction: str,
        entity_map: dict[str, str],
        predicate_map: dict[str, str],
        expand_uris: bool = True,
    ) -> str:
        """
        Replace wd:Label -> QID, and {prefix}:Label -> PID for every
        occurrence of that label under any PREDICATE_PREFIXES prefix.
        """
        for label, qid in entity_map.items():
            replacement = f"<{_WD_NS}{qid}>"
            prediction = prediction.replace(f"wd:{label}", replacement)

        for label, pid in predicate_map.items():
            for prefix in self.PREDICATE_PREFIXES:
                token = f"{prefix}:{label}"
                if token not in prediction:
                    continue
                base_uri = self.COMMON_PREFIXES[prefix]
                replacement = f"<{base_uri}{pid}>"
                prediction = prediction.replace(token, replacement)

        if expand_uris:
            # Expand any remaining prefix:local tokens (rdf:, rdfs:, xsd:, wikibase:, ...).
            skip = {"wd", *self.PREDICATE_PREFIXES}
            for prefix, base_uri in self.COMMON_PREFIXES.items():
                if prefix in skip:
                    continue
                pattern = re.compile(rf"(?<!<)\b{re.escape(prefix)}:([A-Za-z0-9_\-\.]+)")
                prediction = pattern.sub(lambda m, b=base_uri: f"<{b}{m.group(1)}>", prediction)

        return prediction