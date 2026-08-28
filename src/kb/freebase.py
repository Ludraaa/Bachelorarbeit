import re

from src.kb.base import BaseKB


_FB_NS = "http://rdf.freebase.com/ns/"

# MID  : m.0f8l9c  /  g.119pgc8
_MID_RE = re.compile(r"^http://rdf\.freebase\.com/ns/([mg]\.[0-9a-z_]+)$")

# Relation: domain.type.property (>=3 dot-segments, excludes MIDs)
_REL_RE = re.compile(
    r"^http://rdf\.freebase\.com/ns/"
    r"(?!(?:m|g)\.)"
    r"([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){2,})$"
)

_Q_IN_SEXPR = re.compile(r"<(http://rdf\.freebase\.com/ns/[mg]\.[0-9a-z_]+)>")
_P_IN_SEXPR = re.compile(
    r"<(http://rdf\.freebase\.com/ns/"
    r"(?!(?:m|g)\.)"
    r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){2,})>"
)

_PREFIX_RE = re.compile(r"\bfb(p)?:\s*")

# Freebase-only: determines which candidate URIs are Freebase *types*.
# Not part of the BaseKB interface -- Wikidata has no equivalent.
TYPES_QUERY = """
SELECT ?uri WHERE {{
    VALUES ?uri {{ {values} }}
    ?uri <http://rdf.freebase.com/ns/type.object.type>
         <http://rdf.freebase.com/ns/type.type> .
}}
"""


class Freebase(BaseKB):

    KB_PREFIXES = {
        "fb":  _FB_NS,
        "fbp": _FB_NS,
        "ns":  _FB_NS,
    }

    LABEL_QUERY = """
        SELECT ?uri ?label WHERE {{
            VALUES ?uri {{ {values} }}
            ?uri <http://rdf.freebase.com/ns/type.object.name> ?label .
            FILTER(LANG(?label) = "{language}" || LANG(?label) = "")
        }}
    """

    ENTITY_PATTERN = _Q_IN_SEXPR
    RELATION_PATTERN = _P_IN_SEXPR
    ANSWER_URI_PATTERNS = [_MID_RE, _REL_RE]

    TYPES_QUERY = TYPES_QUERY

    # -- Freebase-only extra ------------------------------------------------

    def parse_type_results(self, bindings: list[dict]) -> set[str]:
        """Return the set of URIs from *bindings* that are Freebase types."""
        return {row["uri"]["value"] for row in bindings if "uri" in row}

    # -- normalize ------------------------------------------------------------

    def normalize(self, uri: str) -> str | None:
        """
        Entity MIDs already carry their label (type.object.name), so they are
        returned as-is.

        Relation URIs have no label triple in Freebase; returning None tells
        insert_labels.fetch_labels() to write cache[uri] = None immediately and
        skip the SPARQL batch. format_label() then derives their display token
        from the URI local name.
        """
        if _MID_RE.match(uri):
            return uri
        return None

    # -- parse_label_results ----------------------------------------------------

    def parse_label_results(self, bindings: list[dict]) -> dict[str, str]:
        preferred: dict[str, str] = {}   # uri -> en label
        fallback: dict[str, str] = {}    # uri -> first non-en label

        for row in bindings:
            uri = row.get("uri", {}).get("value", "")
            label = row.get("label", {}).get("value", "")
            lang = row.get("label", {}).get("xml:lang", "")
            if not uri or not label:
                continue
            if lang == "en":
                preferred[uri] = label
            elif uri not in fallback:
                fallback[uri] = label

        return {**fallback, **preferred}  # preferred overwrites fallback

    # -- format_label -------------------------------------------------------------

    def format_label(self, uri: str, label: str) -> str:
        """
        Entity MID  -> fb:Human_Readable_Label
                       (falls back to fb:m.0f8l9c if no name was found)
        Relation    -> fbp:domain.type.property
        """
        if _MID_RE.match(uri):
            local = uri[len(_FB_NS):]
            slug = label.replace(" ", "_") if label else local
            return f"fb:{slug}"

        if _REL_RE.match(uri):
            local = uri[len(_FB_NS):]
            return f"fbp:{local}"

        return self._format_via_common_prefixes(uri)

    # -- format_relation_label ------------------------------------------------------

    def format_relation_label(self, uri: str, label: str) -> str | None:
        if _REL_RE.match(uri):
            return uri[len(_FB_NS):]
        return None

    # -- extract_from_prediction ------------------------------------------------------

    @staticmethod
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

    def extract_from_prediction(self, prediction: str) -> tuple[list[str], list[str]]:
        entities, predicates = [], []
        for m in _PREFIX_RE.finditer(prediction):
            is_pred = m.group(1) is not None
            token = self._scan_token(prediction, m.end()).rstrip(",")
            if not token:
                continue
            (predicates if is_pred else entities).append(token)

        return list(dict.fromkeys(entities)), list(dict.fromkeys(predicates))

    # -- substitute -------------------------------------------------------------------

    def substitute(
        self,
        prediction: str,
        entity_map: dict[str, str],
        predicate_map: dict[str, str],
        expand_uris: bool = True,
    ) -> str:
        """
        Replace fb:Label -> MID and fbp:path -> relation path, then optionally
        expand all remaining prefix:local tokens to full URIs.

        NB divergence from wikidata.py: here the entity_map/predicate_map
        replacements are ALWAYS expanded to full <...> URIs regardless of
        expand_uris (which only governs the generic COMMON_PREFIXES pass
        below). wikidata.py instead makes its wd:/wdt: replacements
        themselves conditional on expand_uris. Preserved as-is from the
        original modules rather than unified, since it's not clear whether
        this was intentional.
        """
        for label, mid in entity_map.items():
            replacement = f"<{_FB_NS}{mid}>"
            prediction = prediction.replace(f"fb:{label}", replacement)

        for label, rel in predicate_map.items():
            replacement = f"<{_FB_NS}{rel}>"
            prediction = prediction.replace(f"fbp:{label}", replacement)

        if expand_uris:
            # Expand any remaining prefix:local tokens (rdf:, rdfs:, xsd:, ...)
            # fb/fbp were already handled above and should stay as-is.
            for prefix, base_uri in self.COMMON_PREFIXES.items():
                if prefix in ("fb", "fbp"):
                    continue
                pattern = re.compile(rf"(?<!<)\b{re.escape(prefix)}:([A-Za-z0-9_\-\.]+)")
                prediction = pattern.sub(lambda m, b=base_uri: f"<{b}{m.group(1)}>", prediction)

        return prediction