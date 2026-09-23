import re

from src.kb.base import BaseKB

# Full Freebase IRI base
_FB_NS = "http://rdf.freebase.com/ns/"

# MID REGEX: m.0f8l9c  /  g.119pgc8 / ...
_MID_RE = re.compile(r"^http://rdf\.freebase\.com/ns/([mg]\.[0-9a-z_]+)$")

# Relation: domain.type.property (minimum of 3 segments; excludes MIDs)
_REL_RE = re.compile(
    r"^http://rdf\.freebase\.com/ns/"
    r"(?!(?:m|g)\.)"
    r"([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){2,})$"
)

# Entity pattern in training format
_Q_IN_SEXPR = re.compile(r"<(http://rdf\.freebase\.com/ns/[mg]\.[0-9a-z_]+)>")

# Relation pattern in training format
_P_IN_SEXPR = re.compile(
    r"<(http://rdf\.freebase\.com/ns/"
    r"(?!(?:m|g)\.)"
    r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){2,})>"
)

# REGEX to capture entity and predicate mentions from prediction beams
_PREFIX_RE = re.compile(r"\bfb(p)?:\s*")


class Freebase(BaseKB):

    # Prefixes used by this module. 'fb' and 'fbp' are used 
    # similar to 'wd' and 'wdt' to distinguish entities from predicates.
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

    TYPES_QUERY = """
        SELECT ?uri WHERE {{
            VALUES ?uri {{ {values} }}
            ?uri <http://rdf.freebase.com/ns/type.object.type>
                 <http://rdf.freebase.com/ns/type.type> .
        }}
    """

    ENTITY_PATTERN = _Q_IN_SEXPR
    RELATION_PATTERN = _P_IN_SEXPR
    ANSWER_URI_PATTERNS = [_MID_RE, _REL_RE]

    # ---------------------------------------------------------------------------
    # Label insertion

    def normalize(self, uri: str) -> str | None:
        """
        Entity MIDs already carry their label (type.object.name).

        Relation URIs have no label triple in Freebase. Return None to skip the
        SPARQL batch in insert_labels.py. format_label() derives the label from 
        the IRI itself.
        """
        if _MID_RE.match(uri):
            return uri
        return None


    def parse_label_results(self, bindings: list[dict]) -> dict[str, str]:
        """
        Prefers english labels. If no english label is found, use the first
        other label instead.
        """
        preferred: dict[str, str] = {}   # IRI -> en label
        fallback: dict[str, str] = {}    # IRI -> first non-en label

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

        return {**fallback, **preferred}


    def format_label(self, uri: str, label: str) -> str:
        """
        Formats label-enriched training data in the following schema:
        
        Entity MID  -> fb:Human_Readable_Label
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


    def format_relation_label(self, uri: str, label: str) -> str | None:
        if _REL_RE.match(uri):
            return uri[len(_FB_NS):]
        return None

    # ---------------------------------------------------------------------------
    # Mention extraction and substitution (resolve step)

    @staticmethod
    def _scan_token(s: str, start: int) -> str:
        """
        Scan a mention while tolerating balanced parentheses, as they can appear in labels.
        Scanning stops at whitespace, semicolon or unmatched closing parenthesis.
        """
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
                # Unbalanced close
                break
            i += 1
        return s[start:i]


    def extract_from_prediction(self, prediction: str) -> tuple[list[str], list[str]]:
        """
        Returns the sets of entity and predicate mentions as a tuple. Searches starting from
        'fb:' and 'fbp:' until whitespace, semicolon or unbalanced parenthesis appears.
        """
        entities, predicates = [], []
        for m in _PREFIX_RE.finditer(prediction):
            is_pred = m.group(1) is not None
            token = self._scan_token(prediction, m.end()).rstrip(",")
            if not token:
                continue
            (predicates if is_pred else entities).append(token)

        return list(dict.fromkeys(entities)), list(dict.fromkeys(predicates))


    def substitute(
        self,
        prediction: str,
        entity_map: dict[str, str],
        predicate_map: dict[str, str],
        expand_uris: bool = True,
    ) -> str:
        """
        Replaces fb:Label -> MID and fbp:path -> relation path, then
        expands all remaining {prefix}:{local} tokens to full IRIs.
        """
        for label, mid in entity_map.items():
            replacement = f"<{_FB_NS}{mid}>"
            prediction = prediction.replace(f"fb:{label}", replacement)

        for label, rel in predicate_map.items():
            replacement = f"<{_FB_NS}{rel}>"
            prediction = prediction.replace(f"fbp:{label}", replacement)

        if expand_uris:
            # Expand any remaining prefix:local tokens (like 'rdf:..' or 'rdfs:..')
            # fb/fbp were already handled above
            for prefix, base_uri in self.COMMON_PREFIXES.items():
                if prefix in ("fb", "fbp"):
                    continue
                pattern = re.compile(rf"(?<!<)\b{re.escape(prefix)}:([A-Za-z0-9_\-\.]+)")
                prediction = pattern.sub(lambda m, b=base_uri: f"<{b}{m.group(1)}>", prediction)

        return prediction