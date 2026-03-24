import json
import re
import subprocess
from collections import defaultdict
from sexpdata import loads
from pprint import pprint
from collections import deque
import json


import re

def fix_sparql_for_jena(sparql: str) -> str:
    """
    Fix common SPARQL issues for Jena parsing:
    - Replace OR with ||
    - Wrap each || term in parentheses inside FILTER
    - Fix xsd:datetime() -> xsd:dateTime()
    - Add PREFIX xsd if missing
    - Fix EXISTS / NOT EXISTS clauses with inner FILTER(xsd:dateTime(...))
    """
    # Flatten whitespace
    sparql_fixed = ' '.join(sparql.split())

    # 1. Replace OR with ||
    sparql_fixed = re.sub(r'\s+OR\s+', ' || ', sparql_fixed, flags=re.IGNORECASE)

    # 2. Wrap each || term in parentheses inside FILTER
    def filter_or_replacer(match):
        inner = match.group(1).strip()
        parts = [p.strip() for p in inner.split('||')]
        wrapped = ' || '.join(f'({p})' for p in parts)
        return f'FILTER ({wrapped})'
    sparql_fixed = re.sub(r'FILTER\s*\((.*?)\)', filter_or_replacer, sparql_fixed)

    # 3. Fix xsd:datetime() -> xsd:dateTime()
    sparql_fixed = re.sub(r'\bxsd:datetime\s*\(', 'xsd:dateTime(', sparql_fixed, flags=re.IGNORECASE)

    # Add common Wikidata prefixes if missing
    common_prefixes = {
        "xsd": "<http://www.w3.org/2001/XMLSchema#>",
        "wd": "<http://www.wikidata.org/entity/>",
        "wdt": "<http://www.wikidata.org/prop/direct/>",
        "p": "<http://www.wikidata.org/prop/>",
        "ps": "<http://www.wikidata.org/prop/statement/>",
        "pq": "<http://www.wikidata.org/prop/qualifier/>",
        "rdfs": "<http://www.w3.org/2000/01/rdf-schema#>",
        "wikibase": "<http://wikiba.se/ontology#>"
    }

    for prefix, uri in common_prefixes.items():
        if f"PREFIX {prefix}:" not in sparql_fixed:
            sparql_fixed = f"PREFIX {prefix}: {uri}\n" + sparql_fixed
   
    return sparql_fixed

### -------------- actual sexpr parsing part
from sexpdata import Symbol

# convert symbol to string if needed
def normalize(x):
    if isinstance(x, Symbol):
        return x.value()
    return x

def is_variable(x):
    return isinstance(x, str) and x.startswith("?")

def extract_select_var(ast):
    if normalize(ast[0]) == 'project':
        return normalize(ast[1][0])

    for child in ast:
        if isinstance(child, list):
            res = extract_select_var(child)
            if res:
                return res

def extract_all_triples(node):
    triples = []

    node = normalize(node)

    if isinstance(node, list):
        if len(node) > 0 and normalize(node[0]) == 'triple':
            s, p, o = node[1:4]
            triples.append((normalize(s), normalize(p), normalize(o)))

        for child in node:
            triples.extend(extract_all_triples(child))

    return triples

def clean_term(x, prefix_map=None):
    x = normalize(x)

    if isinstance(x, str) and x.startswith("<") and x.endswith(">"):
        iri = x[1:-1]

        if prefix_map:
            for prefix, ns in prefix_map.items():
                if iri.startswith(ns):
                    return f"{prefix}:{iri[len(ns):]}"

        return iri  # fallback: full IRI

    return x

def build_graph(triples):
    # use set to avoid duplicates
    graph = defaultdict(set)

    for s, p, o in triples:
        s = clean_term(s)
        p = clean_term(p)
        o = clean_term(o)

        # s → o
        graph[s].add((p, o, "subject"))

        # o → s
        graph[o].add((p, s, "object"))

    # return as list
    return {k: list(v) for k, v in graph.items()}


def build_sexpr(graph, target):
    """
    Traverse the graph from `target` variable.
    Returns a nested dict structure suitable for SExpr generation
    that emulates Jena's nested AND behavior.
    """

    def dfs(node, visited_nodes):
        visited_nodes.add(node)
        edges = graph.get(node, [])

        # Leaf: no outgoing edges or all children visited
        if not edges or all(is_variable(other) and other in visited_nodes for _, other, _ in edges):
            return []

        children_exprs = []

        for pred, other, direction in edges:
            if is_variable(other) and other in visited_nodes:
                continue

            edge_tuple = (node, pred, other, direction)

            if not is_variable(other):
                # Constant leaf
                child_expr = [edge_tuple]
                # Continue DFS in case siblings exist further
                sub_exprs = dfs(other, visited_nodes.copy())
                if sub_exprs:
                    # Combine child path with its subtrees
                    for s in sub_exprs:
                        children_exprs.append(child_expr + s)
                else:
                    children_exprs.append(child_expr)
            else:
                # Variable child
                sub_exprs = dfs(other, visited_nodes.copy())
                if sub_exprs:
                    for s in sub_exprs:
                        children_exprs.append([edge_tuple] + s)
                else:
                    children_exprs.append([edge_tuple])

        return children_exprs

    all_paths = dfs(target, set())

    # Deduplicate top-level paths
    seen = set()
    unique_paths = []
    for p in all_paths:
        key = tuple(p)
        if key not in seen:
            unique_paths.append(p)
            seen.add(key)

    return {target: unique_paths}

def paths_to_sexpr(logical_paths):
    """
    Convert logical paths to SExpr with:
    - Common prefix factoring
    - Right-associative AND (atomic/simple paths on right)
    - Path ordering heuristics: constants first, more R relations right
      applied at every combination step
    """

    all_paths = list(logical_paths.values())[0]

    def is_variable_name(x):
        return x.startswith('?')

    # Heuristic for path ordering
    def path_priority(path):
        const_count = sum(1 for s, p, o, d in path if not is_variable_name(s) or not is_variable_name(o))
        r_count = sum(1 for s, p, o, d in path if d == "object")
        return (-const_count, r_count, len(path))  # more constants left, more R relations right

    def path_to_join(path, suffix=None):
        """Convert a single path (list of edges) to nested JOIN."""
        expr = suffix
        for s, p, o, direction in reversed(path):
            pred_str = f"(R {p})" if direction == "object" else p
            other = o if direction == "subject" else s
            if is_variable_name(other):
                candidate = s if other != s else o
                if not is_variable_name(candidate):
                    other = candidate
            if expr is None:
                expr = other
            expr = f"(JOIN {pred_str} {expr})"
        return expr

    def combine_paths(paths):
        if not paths:
            return ""
        # Sort paths at every recursive step according to importance heuristic
        paths = sorted(paths, key=path_priority)

        if len(paths) == 1:
            return path_to_join(paths[0])

        # Factor common prefix
        min_len = min(len(p) for p in paths)
        prefix_len = 0
        for i in range(min_len):
            first_edge = paths[0][i]
            if all(len(p) > i and p[i] == first_edge for p in paths):
                prefix_len += 1
            else:
                break

        if prefix_len > 0:
            prefix = paths[0][:prefix_len]
            suffixes = [p[prefix_len:] for p in paths if p[prefix_len:]]
            if not suffixes:
                return path_to_join(prefix)
            combined_suffix = combine_paths(suffixes)
            return path_to_join(prefix, suffix=combined_suffix)
        else:
            # No common prefix: right-associative AND
            first, *rest = paths
            inner = combine_paths(rest)
            return f"(AND {path_to_join(first)} {inner})"

    # Initial top-level combination
    return combine_paths(all_paths)

import re

# just for comparison to their sexpr for now
def strip_prefixes(sexpr: str) -> str:
    """
    Strips the Freebase prefix from an SExpr string.
    Keeps constants (m.*) intact.
    """
    # Regex to match full Freebase URLs
    pattern = r"http://rdf\.freebase\.com/ns/([a-zA-Z0-9_.]+)"
    
    # Replace with just the namespace/predicate part
    stripped = re.sub(pattern, r"\1", sexpr)
    return stripped


# Load dataset
#json_file_path = "data/CWQ/sexpr/CWQ.train.expr.json"
json_file_path = "data/wdql_test1k.json"

with open(json_file_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

for entry in dataset:
    #question_id = entry.get("ID", "UNKNOWN")
    sparql_query = entry.get("sparql", "").strip()
    sparql_query_fixed = fix_sparql_for_jena(sparql_query)
    #sparql_query_fixed = sparql_query
    saved_sexpr = entry.get("SExpr", "N/A")
    
    question_id = entry.get("id", "Unknown")
    

    print(f"\n=== Question ID: {question_id} ===")
    print("Question:")
    print(entry.get("question", "").strip(), entry.get("utterance", "").strip())
    print()
    print("SPARQL:")
    print(sparql_query)
    print()

    try:
        result = subprocess.run(
        [
            "./ApacheJena/java/jdk-21.0.10/bin/java",
            "-cp",
            "ApacheJena:ApacheJena/apache-jena-6.0.0/lib/*",
            "SparqlToSExpr",
            sparql_query_fixed
        ],
        capture_output=True,
        text=True,
        timeout=30
    )

        # If Jena failed, show stderr and skip
        if result.returncode != 0:
            print(f"Jena failed with return code {result.returncode}.")
            if result.stderr.strip():
                print("=== Jena Errors (STDERR) ===")
                print(result.stderr)
            print()
            continue

        # Only print stdout if Jena succeeded
        res = result.stdout
        print("=== Jena Output (STDOUT) ===")
        print(res)
        print()

        # Parse AST if possible
        try:
            ast = loads(res)
            print("AST parsed:")
            pprint(ast, width=120)
            print()
        except Exception as e:
            print(f"Failed to parse AST: {e}, skipping.\n")
            continue

    except Exception as e:
        print(f"Jena execution failed: {e}, skipping.\n")
        continue

    print("Graph:")
    print("[")
    triples = extract_all_triples(ast)
    graph = build_graph(triples)
    for node, edges in graph.items():
        print(f"\n\t{node}")
        for pred, other, direction in edges:
            print(f"\t\t{direction} --[{pred}]--> {other}")
    print("]")
    print()

    print("Target extraction:")
    target = extract_select_var(ast)
    print(target)
    print()

    print("SExpr construction:")
    sexpr_schema = build_sexpr(graph, target)
    print(sexpr_schema)
    print()
    sexpr = strip_prefixes(paths_to_sexpr(sexpr_schema))
    print("Final SExpr:")
    print(sexpr)
    print()

    print("Saved SExpr:")
    print(saved_sexpr)
    print()


    input("\n\n")
