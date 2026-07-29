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

def fix_sparql_for_jena(sparql: str, COMMON_PREFIXES) -> str:
    """
    Normalize SPARQL so Jena can parse it reliably:
    - remove inline comments
    - normalize whitespace
    - fix small syntax quirks
    - inject missing PREFIX declarations
    """

    # remove comments safely
    def remove_comments(s: str) -> str:
        result = []
        in_string = False
        quote = None
        i = 0

        while i < len(s):
            c = s[i]

            if c in ('"', "'"):
                if not in_string:
                    in_string = True
                    quote = c
                elif quote == c:
                    in_string = False
                result.append(c)
                i += 1
                continue

            if not in_string and c == "#":
                while i < len(s) and s[i] != "\n":
                    i += 1
                continue

            result.append(c)
            i += 1

        return "".join(result)

    sparql = remove_comments(sparql)

    # collapse whitespace
    sparql = " ".join(sparql.split())

    # fix common SPARQL quirks
    sparql = re.sub(r"\s+OR\s+", " || ", sparql, flags=re.IGNORECASE)
    sparql = re.sub(r"\bxsd:datetime\s*\(", "xsd:dateTime(", sparql, flags=re.IGNORECASE)

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