import os
import re
import subprocess
import threading
import base64

# ---------------------------------------------------------------------------
# Environment setup

_JAVA_HOME = os.environ.get("JAVA_HOME")
_JENA_HOME = os.environ.get("JENA_HOME")

if not _JAVA_HOME:
    raise ValueError("JAVA_HOME environment variable is not set.")

if not _JENA_HOME:
    raise ValueError("JENA_HOME environment variable is not set.")

JAVA = os.path.join(_JAVA_HOME, "bin", "java")
JENA_DIR = os.path.join(os.getcwd(), "src", "sexpr")
JENA_SERVER_CLASS = os.path.join(JENA_DIR, "JenaServer.class")

if not os.path.isfile(JAVA):
    raise FileNotFoundError(
        f"Java executable not found at {JAVA}. Verify JAVA_HOME is set correctly."
    )

if not os.path.isfile(JENA_SERVER_CLASS):
    raise FileNotFoundError(
        f"JenaServer.class not found at {JENA_SERVER_CLASS}. Compile JenaServer.java first."
    )

CLASSPATH = (
    os.path.join(_JENA_HOME, "lib", "*")
    + ":"
    + JENA_DIR
)

# ---------------------------------------------------------------------------
# JVM process

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


_bridge = JenaBridge()


def sparql_to_algebra(sparql: str) -> str:
    """
    Transforms a SPARQL query to Jena Syntax Expression.
    """
    return _bridge.request(
        "sparql_to_algebra",
        sparql
    )


def algebra_to_sparql(algebra: str) -> str:
    """
    Transforms a Jena Syntax Expression into a SPARQL query.
    """
    return _bridge.request(
        "algebra_to_sparql",
        algebra
    )

# ---------------------------------------------------------------------------
# Query preprocessing

# These are needed because the Jena parser is very strict.
# These preprocessing steps are not perfect and pretty ad-hoc, but they
# fix most common errors.

def remove_comments(s: str) -> str:
    """
    Walks through the characters of the query and reassembles the query
    without comments.
    """
    result = []
    in_string = False
    quote = None
    in_iri = False
    i = 0

    # Iterate through characters
    while i < len(s):
        c = s[i]

        # Enter string
        if not in_iri and c in ('"', "'"):
            if not in_string:
                in_string = True
                quote = c
            elif quote == c:
                in_string = False
            result.append(c)
            i += 1
            continue

        # Enter IRI
        if not in_string and c == "<":
            in_iri = True
            result.append(c)
            i += 1
            continue

        # Exit IRI
        if in_iri and c == ">":
            in_iri = False
            result.append(c)
            i += 1
            continue

        # Enter comment!
        if not in_string and not in_iri and c == "#":
            while i < len(s) and s[i] != "\n":
                i += 1
            continue

        result.append(c)
        i += 1
        
    # Reassemble query
    return "".join(result)


def _mask_literals(s: str) -> str:
    """
    Return a copy of a given query where the contents of string literals and IRIs are
    blanked out to whitespace.
    """
    out = []
    in_string = False
    quote = None
    in_iri = False
    i = 0
    n = len(s)

    # Iterate through characters
    while i < n:
        c = s[i]

        # Enter string
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

        # Enter IRI
        if not in_string and c == "<":
            in_iri = True
            out.append(c)
            i += 1
            continue

        # Exit IRI
        if in_iri and c == ">":
            in_iri = False
            out.append(c)
            i += 1
            continue

        # While in string or IRI, write whitespace
        if in_string or in_iri:
            out.append(' ')
        else:
            out.append(c)
        i += 1

    # Reassemble query
    return "".join(out)


def _collapse_whitespace_outside_literals(s: str) -> str:
    """
    Iterates over characters and collapses multiple consecutive whitespaces into single ones.
    Ignores string literals.
    """

    out = []
    in_string = False
    quote = None
    in_iri = False
    pending_space = False
    i = 0
    n = len(s)

    # Iterate through characters
    while i < n:
        c = s[i]

        # Enter string
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

        # Enter IRI
        if not in_string and c == "<":
            if pending_space:
                out.append(' ')
                pending_space = False
            in_iri = True
            out.append(c)
            i += 1
            continue

        # Exit IRI
        if in_iri and c == ">":
            in_iri = False
            out.append(c)
            i += 1
            continue

        # In string or IRI, write back regularly
        if in_string or in_iri:
            out.append(c)
            i += 1
            continue

        # Outside of string or IRI, remove whitespace
        if c.isspace():
            pending_space = bool(out)  # don't emit a leading space
            i += 1
            continue
        
        # Potentially add back whitespace
        if pending_space:
            out.append(' ')
            pending_space = False
        out.append(c)
        i += 1

    # Reassemble query
    return "".join(out)


# Aggregation regex
_AGG_RE = re.compile(
    r'\(\s*(?:COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|SAMPLE)\b', re.IGNORECASE
)
# SELECT regex
_SELECT_RE = re.compile(r'\bSELECT\b(?:\s+DISTINCT\b|\s+REDUCED\b)?', re.IGNORECASE)


def _plain_vars_at_depth0(select_list_masked: str) -> list[str]:
    """
    Returns a list of all directly projected SELECT vars. This excludes 
    variables that appear inside an aggregate expression.
    """
    vars_found = []
    depth = 0
    i = 0
    n = len(select_list_masked)
    # Iterate over characters
    while i < n:
        c = select_list_masked[i]
        # Track paranthesis depth
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        # Found directly projected variable
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
    Add the 'GROUP BY' that Jena requires for aggregates.
    """
    # Mask sparql to make depth tracking safe
    masked = _mask_literals(sparql)

    out = []
    i = 0
    n = len(sparql)

    # Iterate through characters
    while i < n:
        m = _SELECT_RE.search(masked, i)
        
        # No SELECT found
        if not m:
            out.append(sparql[i:])
            break

        # SELECT found
        out.append(sparql[i:m.end()])
        sel_start = m.end()

        # Advance to the "{" that opens this SELECT's WHERE block
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
        # Mask literals and IRIs
        select_list_masked = masked[sel_start:brace_pos]

        # Drop a trailing "WHERE" keyword from the select list
        strip_where = re.search(r'\bWHERE\s*$', select_list_masked, re.IGNORECASE)
        if strip_where:
            select_list = select_list[:strip_where.start()]
            select_list_masked = select_list_masked[:strip_where.start()]

        # Find the brace matching brace_pos (WHERE block of this SELECT)
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

        # Fix any nested SELECTs inside this WHERE block first
        inner = sparql[brace_pos + 1:close_pos]
        inner_fixed = _add_missing_group_by(inner)
        where_block_fixed = "{" + inner_fixed + "}"

        # Contains aggregate
        has_agg = _AGG_RE.search(select_list_masked) is not None
        if has_agg:
            plain_vars = _plain_vars_at_depth0(select_list_masked)
            after_masked = masked[close_pos + 1:]
            already_grouped = re.match(r'\s*GROUP\s+BY\b', after_masked, re.IGNORECASE) is not None

            # Add GROUP BY
            if plain_vars and not already_grouped:
                where_block_fixed += " GROUP BY " + " ".join(dict.fromkeys(plain_vars))

        out.append(select_list)
        out.append(where_block_fixed)
        i = close_pos + 1

    # Reassemble query
    return "".join(out)


def fix_sparql_for_jena(sparql: str, COMMON_PREFIXES) -> str:
    """
    Normalizes SPARQL so Jena can parse it (more) reliably:
    - Removes inline comments
    - Normalize whitespace
    - Fix other small syntax quirks
    - Fix a specific gold data bug where aggregate + plain var is in SELECT with no GROUP BY
    - Inject missing PREFIX declarations
    """

    sparql = remove_comments(sparql)

    # Collapse whitespace 
    sparql = _collapse_whitespace_outside_literals(sparql)

    # Fix common SPARQL quirks
    sparql = re.sub(r"\s+OR\s+", " || ", sparql, flags=re.IGNORECASE)
    sparql = re.sub(r"\bxsd:datetime\s*\(", "xsd:dateTime(", sparql, flags=re.IGNORECASE)

    # Fix potentially missing GROUP BY on SELECT clauses
    sparql = _add_missing_group_by(sparql)

    # Inject missing prefixes (just inject all we define for now)
    for prefix, uri in COMMON_PREFIXES.items():
        if f"PREFIX {prefix}:" not in sparql:
            sparql = f"PREFIX {prefix}: <{uri}>\n" + sparql

    return sparql


PREFIX_BLOCK_RE = re.compile(r'^\(prefix\s*\(\(.*?\)\)\s*', re.DOTALL)

def strip_prefix_and_expand(algebra: str, common_prefixes: dict[str, str]) -> str:
    """
    Removes the prefix declarations at the top of the syntax expression and expands all
    occurances to use full IRI instead of prefix.
    """
    # Remove prefix block
    had_prefix = algebra.lstrip().startswith("(prefix")
    algebra = PREFIX_BLOCK_RE.sub("", algebra)

    # Strip closing parantheses
    if had_prefix:
        algebra = algebra.rstrip()
        if algebra.endswith(")"):
            algebra = algebra[:-1]

    # Expand prefixes
    for prefix, base_uri in common_prefixes.items():
        pattern = re.compile(
            rf'(?<!<)\b{re.escape(prefix)}:([A-Za-z0-9_\-\.]+)'
        )
        algebra = pattern.sub(
            lambda m, base=base_uri: f"<{base}{m.group(1)}>",
            algebra
        )

    return algebra



def detect_query_form(raw_sparql: str) -> str:
    """
    Jena converts always produces SELECT queries from syntax expressions.
    This function detects the query form of the query. This is later used to re-inject.
    """

    # Normalize whitespace
    q = " ".join(raw_sparql.split())

    # Remove PREFIX blocks completely
    q = re.sub(r"PREFIX\s+\w+:\s*<[^>]*>", "", q, flags=re.IGNORECASE)

    q = q.strip().upper()

    # Look for ASK keyword
    if re.match(r"ASK\b", q):
        return "ASK"

    # Look for SELECT keyword
    if re.match(r"SELECT\b", q):
        return "SELECT"

    # Unsupported
    raise ValueError(f"Unknown Query Form for query:\n{raw_sparql}")


def restore_query_form(form: str, sparql: str) -> str:
    """
    Re-injects the correct query form into the query produced by Jena.
    """
    # Was SELECT
    if form != "ASK":
        return sparql

    sparql = sparql.strip()

    # Find WHERE block
    match = re.search(r"WHERE\s*\{", sparql, re.IGNORECASE)
    # No WHERE block, should never happen
    if not match:
        return sparql  

    where_start = match.start()

    # Discard everything up to WHERE block
    where_clause = sparql[where_start:]

    return "ASK " + where_clause