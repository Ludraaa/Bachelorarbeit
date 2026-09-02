import os
import re
import json
import atexit
import subprocess
import threading
import base64

# ---------------------------------------------------------------------------
# Environment setup

_JAVA_HOME = os.environ.get("JAVA_HOME", "")
_JENA_HOME = os.environ.get("JENA_HOME", "")

JAVA = os.path.join(_JAVA_HOME, "bin", "java")

CLASSPATH = (
    os.path.join(_JENA_HOME, "lib", "*")
    + ":"
    + os.path.join(os.getcwd(), "ApacheJena")
)

# ---------------------------------------------------------------------------
# Persistent JVM process

class JenaBridge:

    def __init__(self):

        self.proc = subprocess.Popen(
            [JAVA, "-cp", CLASSPATH, "JenaServer"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self.lock = threading.Lock()

    def close(self):

        if self.proc.poll() is None:
            self.proc.kill()

    def request(self, cmd: str, data: str) -> str:

        with self.lock:

            encoded = base64.b64encode(
                data.encode("utf-8")
            ).decode("ascii")

            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.write(encoded + "\n")
            self.proc.stdin.flush()

            status = self.proc.stdout.readline().strip()
            payload = self.proc.stdout.readline()

            if not payload:
                raise RuntimeError("Jena server died")

            payload = payload.strip()

            decoded = base64.b64decode(payload).decode("utf-8")

            if status == "ERR":
                raise RuntimeError(decoded)

            return decoded


# ---------------------------------------------------------------------------
# Public API

_bridge = JenaBridge()


def sparql_to_algebra(sparql: str) -> str:
    return _bridge.request(
        "sparql_to_algebra",
        sparql
    )


def algebra_to_sparql(algebra: str) -> str:
    return _bridge.request(
        "algebra_to_sparql",
        algebra
    )

# ---------------------------------------------------------------------------
# other helpers

def remove_comments(s: str) -> str:
    result = []
    in_string = False
    quote = None
    in_iri = False
    i = 0

    while i < len(s):
        c = s[i]

        if not in_iri and c in ('"', "'"):
            if not in_string:
                in_string = True
                quote = c
            elif quote == c:
                in_string = False
            result.append(c)
            i += 1
            continue

        if not in_string and c == "<":
            in_iri = True
            result.append(c)
            i += 1
            continue

        if in_iri and c == ">":
            in_iri = False
            result.append(c)
            i += 1
            continue

        if not in_string and not in_iri and c == "#":
            while i < len(s) and s[i] != "\n":
                i += 1
            continue

        result.append(c)
        i += 1

    return "".join(result)


def _mask_literals(s: str) -> str:
    """
    Return a same-length copy of *s* with the contents of string literals
    and IRIs (<...>) blanked out to spaces, so bracket/paren/keyword
    scanning elsewhere isn't confused by punctuation inside them.
    Positions line up 1:1 with the original string, so callers can search
    the masked text and then slice the ORIGINAL string at the same
    indices. Mirrors the string/IRI tracking in remove_comments() above.
    """
    out = []
    in_string = False
    quote = None
    in_iri = False
    i = 0
    n = len(s)

    while i < n:
        c = s[i]

        if not in_iri and c in ('"', "'"):
            if not in_string:
                in_string = True
                quote = c
                out.append(c)
            elif quote == c:
                in_string = False
                out.append(c)
            else:
                out.append(' ')
            i += 1
            continue

        if not in_string and c == "<":
            in_iri = True
            out.append(c)
            i += 1
            continue

        if in_iri and c == ">":
            in_iri = False
            out.append(c)
            i += 1
            continue

        if in_string or in_iri:
            out.append(' ')
        else:
            out.append(c)
        i += 1

    return "".join(out)


def _collapse_whitespace_outside_literals(s: str) -> str:
    """
    Like " ".join(s.split()), but quote/IRI-aware: collapses runs of
    whitespace to a single space and trims the ends, EXCEPT inside string
    literals or <...> IRIs, whose contents (including any internal
    whitespace, e.g. multiple spaces) are passed through byte-for-byte.

    This matters because a naive global ``" ".join(sparql.split())`` does
    not know it's inside a quoted literal and will happily collapse
    whitespace there too -- silently changing the literal's value. If the
    underlying KB has that exact (unusual) literal stored with e.g. a
    double space, collapsing it to a single space makes the query stop
    matching anything, with no error raised anywhere. Mirrors the
    string/IRI tracking in remove_comments() / _mask_literals() above.
    """
    out = []
    in_string = False
    quote = None
    in_iri = False
    pending_space = False
    i = 0
    n = len(s)

    while i < n:
        c = s[i]

        if not in_iri and c in ('"', "'"):
            if pending_space:
                out.append(' ')
                pending_space = False
            if not in_string:
                in_string = True
                quote = c
            elif quote == c:
                in_string = False
            out.append(c)
            i += 1
            continue

        if not in_string and c == "<":
            if pending_space:
                out.append(' ')
                pending_space = False
            in_iri = True
            out.append(c)
            i += 1
            continue

        if in_iri and c == ">":
            in_iri = False
            out.append(c)
            i += 1
            continue

        if in_string or in_iri:
            out.append(c)
            i += 1
            continue

        # outside any literal/IRI: collapse whitespace runs, trim at ends
        if c.isspace():
            pending_space = bool(out)  # don't emit a leading space
            i += 1
            continue

        if pending_space:
            out.append(' ')
            pending_space = False
        out.append(c)
        i += 1

    return "".join(out)


_AGG_RE = re.compile(
    r'\(\s*(?:COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|SAMPLE)\b', re.IGNORECASE
)
_SELECT_RE = re.compile(r'\bSELECT\b(?:\s+DISTINCT\b|\s+REDUCED\b)?', re.IGNORECASE)


def _plain_vars_at_depth0(select_list_masked: str) -> list[str]:
    """
    Variables projected directly in a SELECT list (depth 0), as opposed to
    ones only appearing inside an aggregate expression's parens (depth >0).
    Per SPARQL grammar, aggregate expressions are always parenthesised
    ('(AGG(...) AS ?v)'), so anything at depth 0 is a plain projected var.
    """
    vars_found = []
    depth = 0
    i = 0
    n = len(select_list_masked)
    while i < n:
        c = select_list_masked[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == '?' and depth == 0:
            j = i + 1
            while j < n and (select_list_masked[j].isalnum() or select_list_masked[j] == '_'):
                j += 1
            vars_found.append(select_list_masked[i:j])
            i = j
            continue
        i += 1
    return vars_found


def _add_missing_group_by(sparql: str) -> str:
    """
    Some CWQ/WebQSP gold queries project an aggregate (COUNT, SUM, ...)
    alongside a plain, non-aggregated variable with no GROUP BY clause,
    e.g. 'SELECT ?x (COUNT(?x) AS ?count) WHERE { ... }' — a common
    "argmax" idiom (per-?x counts, then filtered against a separately
    computed max) where the author forgot GROUP BY ?x. Some SPARQL
    engines accept this leniently; Jena's strict validator rejects it
    with "Non-group key variable in SELECT". This detects the pattern,
    at any nesting depth, and injects the missing GROUP BY.
    """
    masked = _mask_literals(sparql)

    out = []
    i = 0
    n = len(sparql)

    while i < n:
        m = _SELECT_RE.search(masked, i)
        if not m:
            out.append(sparql[i:])
            break

        out.append(sparql[i:m.end()])
        sel_start = m.end()

        # Advance to the "{" that opens this SELECT's WHERE block, i.e.
        # the first unparenthesized "{" after the select list.
        depth_paren = 0
        j = sel_start
        brace_pos = None
        while j < n:
            c = masked[j]
            if c == '(':
                depth_paren += 1
            elif c == ')':
                depth_paren -= 1
            elif c == '{' and depth_paren == 0:
                brace_pos = j
                break
            j += 1

        if brace_pos is None:
            out.append(sparql[sel_start:])
            i = n
            break

        select_list = sparql[sel_start:brace_pos]
        select_list_masked = masked[sel_start:brace_pos]

        # Drop a trailing "WHERE" keyword from the select list, if present.
        strip_where = re.search(r'\bWHERE\s*$', select_list_masked, re.IGNORECASE)
        if strip_where:
            select_list = select_list[:strip_where.start()]
            select_list_masked = select_list_masked[:strip_where.start()]

        # Find the brace matching brace_pos (this SELECT's WHERE block).
        depth_brace = 0
        k = brace_pos
        close_pos = None
        while k < n:
            c = masked[k]
            if c == '{':
                depth_brace += 1
            elif c == '}':
                depth_brace -= 1
                if depth_brace == 0:
                    close_pos = k
                    break
            k += 1

        if close_pos is None:
            out.append(sparql[sel_start:])
            i = n
            break

        # Fix any nested SELECTs inside this WHERE block first.
        inner = sparql[brace_pos + 1:close_pos]
        inner_fixed = _add_missing_group_by(inner)
        where_block_fixed = "{" + inner_fixed + "}"

        has_agg = _AGG_RE.search(select_list_masked) is not None
        if has_agg:
            plain_vars = _plain_vars_at_depth0(select_list_masked)
            after_masked = masked[close_pos + 1:]
            already_grouped = re.match(r'\s*GROUP\s+BY\b', after_masked, re.IGNORECASE) is not None

            if plain_vars and not already_grouped:
                where_block_fixed += " GROUP BY " + " ".join(dict.fromkeys(plain_vars))

        out.append(select_list)
        out.append(where_block_fixed)
        i = close_pos + 1

    return "".join(out)


def fix_sparql_for_jena(sparql: str, COMMON_PREFIXES) -> str:
    """
    Normalize SPARQL so Jena can parse it reliably:
    - remove inline comments
    - normalize whitespace (outside string literals / IRIs -- see
      _collapse_whitespace_outside_literals)
    - fix small syntax quirks
    - fix a specific gold-data bug: aggregate + plain var in SELECT with
      no GROUP BY (rejected by Jena's strict validator; see
      _add_missing_group_by for details)
    - inject missing PREFIX declarations
    """

    sparql = remove_comments(sparql)

    # collapse whitespace -- but never inside string literals or IRIs,
    # since some gold literals have meaningful internal whitespace (e.g.
    # a stored KB literal with a double space) that a naive global collapse
    # would silently alter, changing what the query actually matches.
    sparql = _collapse_whitespace_outside_literals(sparql)

    # fix common SPARQL quirks
    sparql = re.sub(r"\s+OR\s+", " || ", sparql, flags=re.IGNORECASE)
    sparql = re.sub(r"\bxsd:datetime\s*\(", "xsd:dateTime(", sparql, flags=re.IGNORECASE)

    # fix missing GROUP BY on SELECT clauses that mix an aggregate with a
    # plain variable (see _add_missing_group_by docstring)
    sparql = _add_missing_group_by(sparql)

    # inject missing prefixes
    for prefix, uri in COMMON_PREFIXES.items():
        if f"PREFIX {prefix}:" not in sparql:
            sparql = f"PREFIX {prefix}: <{uri}>\n" + sparql

    return sparql



PREFIX_BLOCK_RE = re.compile(r'^\(prefix\s*\(\(.*?\)\)\s*', re.DOTALL)

def strip_prefix_and_expand(algebra: str, common_prefixes: dict[str, str]) -> str:
    # remove prefix block
    had_prefix = algebra.lstrip().startswith("(prefix")
    algebra = PREFIX_BLOCK_RE.sub("", algebra)

    # strip closing parantheses
    if had_prefix:
        algebra = algebra.rstrip()
        if algebra.endswith(")"):
            algebra = algebra[:-1]

    # expand prefixes
    for prefix, base_uri in common_prefixes.items():
        pattern = re.compile(
            rf'(?<!<)\b{re.escape(prefix)}:([A-Za-z0-9_\-\.]+)'
        )
        algebra = pattern.sub(
            lambda m, base=base_uri: f"<{base}{m.group(1)}>",
            algebra
        )

    return algebra

# ---------------------------------------------------------------------------
# for sparql target (jena roundtrip)
# ASK queries get no project layer in jena, but when reconstructing sparql from jena, it auto inserts 'SELECT *'

def detect_query_form(raw_sparql: str) -> str:

    # normalize whitespace early
    q = " ".join(raw_sparql.split())

    # remove PREFIX blocks completely
    q = re.sub(r"PREFIX\s+\w+:\s*<[^>]*>", "", q, flags=re.IGNORECASE)

    q = q.strip().upper()

    # now detect
    if "ASK" in q[:20]:
        return "ASK"

    if q.startswith("SELECT"):
        return "SELECT"

    if q.startswith("CONSTRUCT"):
        raise ValueError(f"Unknown Query Form for query:\n{raw_sparql}")

    if q.startswith("DESCRIBE"):
        raise ValueError(f"Unknown Query Form for query:\n{raw_sparql}")

    raise ValueError(f"Unknown Query Form for query:\n{raw_sparql}")


def restore_query_form(form: str, sparql: str) -> str:
    if form != "ASK":
        return sparql

    sparql = sparql.strip()

    # remove SELECT ... WHERE prefix
    # handles: SELECT *, SELECT ?x ?y, etc.
    match = re.search(r"WHERE\s*\{", sparql, re.IGNORECASE)
    if not match:
        return sparql  # should never happen

    where_start = match.start()

    # keep only WHERE clause onward
    where_clause = sparql[where_start:]

    return "ASK " + where_clause