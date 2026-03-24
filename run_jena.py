import json
import re
import subprocess
from collections import defaultdict
from sexpdata import loads
from pprint import pprint
from collections import deque


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

def extract_select_vars(ast):
    """
    Recursively extract all variables listed in a SPARQL project clause.
    Returns a list of variable names (e.g., ['?person', '?startDate']).
    """
    vars_list = []

    if isinstance(ast, list):
        head = normalize(ast[0])
        if head == 'project' and len(ast) > 1:
            # ast[1] is the list of projected vars
            for v in ast[1]:
                v_norm = normalize(v)
                if is_variable(v_norm):
                    vars_list.append(v_norm)

        # Recurse into children
        for child in ast:
            if isinstance(child, list):
                vars_list.extend(extract_select_vars(child))

    return vars_list


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


def split_into_subgraphs(graph):
    """
    Splits a graph into connected subgraphs.

    Input:
        graph: dict[node] -> list of (pred, other, direction)

    Output:
        list of subgraph dicts (same structure as input graph)
    """

    visited = set()
    subgraphs = []

    for start in graph:
        if start in visited:
            continue

        # BFS to collect one connected component
        queue = deque([start])
        component_nodes = set()

        while queue:
            node = queue.popleft()
            if node in visited:
                continue

            visited.add(node)
            component_nodes.add(node)

            for _, neighbor, _ in graph.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        # Build subgraph from collected nodes
        subgraph = {}
        for node in component_nodes:
            subgraph[node] = [
                (pred, other, direction)
                for (pred, other, direction) in graph.get(node, [])
                if other in component_nodes
            ]

        subgraphs.append(subgraph)

    return subgraphs


def find_object_nodes(graph):
    object_nodes = set()

    for node, edges in graph.items():
        for pred, other, direction in edges:
            if direction == "subject":
                # node --subject--> other
                # => other is object
                object_nodes.add(other)

    return object_nodes

def select_root_by_object_rule(graph):
    """
    Select root for SExpr traversal:
    1. Prefer variables that appear as subjects.
    2. If none, pick constants that are subjects.
    3. Fallback: pick node with minimal eccentricity (most central),
       tie-break on degree and lexicographic order.
    """
    # Build subject sets
    subjects = set()
    for node, edges in graph.items():
        for pred, other, direction in edges:
            if direction == "subject":
                subjects.add(node)

    # Candidate sets
    var_subjects = [n for n in graph.keys() if is_variable(n) and n in subjects]
    const_subjects = [n for n in graph.keys() if not is_variable(n) and n in subjects]

    if var_subjects:
        # Tie-break: highest degree, then lexicographic
        return max(var_subjects, key=lambda n: (len(graph[n]), n))
    if const_subjects:
        return max(const_subjects, key=lambda n: (len(graph[n]), n))

    # Fallback: compute eccentricity
    def eccentricity(node):
        visited = set()
        queue = deque([(node, 0)])
        max_dist = 0
        while queue:
            current, dist = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            max_dist = max(max_dist, dist)
            for _, neighbor, _ in graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return max_dist

    eccs = [(eccentricity(n), len(graph[n]), n) for n in graph.keys()]
    # Minimize eccentricity, then maximize degree, then lexicographic
    eccs.sort(key=lambda x: (x[0], -x[1], x[2]))
    return eccs[0][2]


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

def paths_to_sexpr(logical_paths, return_vars=None, root=None):
    """
    Convert logical paths to SExpr, marking return variables that appear as subjects.
    """

    all_paths = list(logical_paths.values())[0]

    def is_variable_name(x):
        return isinstance(x, str) and x.startswith('?')

    def path_priority(path):
        const_count = sum(1 for s, p, o, d in path if not is_variable_name(s) or not is_variable_name(o))
        r_count = sum(1 for s, p, o, d in path if d == "object")
        return (-const_count, r_count, len(path))

    def path_to_join(path, suffix=None):
        """Builds the SExpr for a single path, adding SUBJECT for vars that are subjects."""
        expr = suffix

        # traverse from leaf to root
        for a, p, b, direction in reversed(path):
            pred_str = f"(R {p})" if direction == "object" else p

            node = b if direction == "subject" else a
            if not expr:
                expr = node

            # if the subject of this edge is a return var, wrap in SUBJECT
            if direction == "subject" and return_vars and a in return_vars and a != root:
                # Wrap the JOIN subtree in SUBJECT
                expr = f"(SUBJECT {a} (JOIN {pred_str} {expr}))"
            else:
                # regular JOIN
                expr = f"(JOIN {pred_str} {expr})"

        return expr

    def combine_paths(paths):
        if not paths:
            return ""

        paths = sorted(paths, key=path_priority)

        if len(paths) == 1:
            return path_to_join(paths[0])

        # Factor common prefix
        min_len = min(len(p) for p in paths)
        prefix_len = 0

        for i in range(min_len):
            edge = paths[0][i]
            if all(len(p) > i and p[i] == edge for p in paths):
                prefix_len += 1
            else:
                break

        if prefix_len > 0:
            prefix = paths[0][:prefix_len]
            suffixes = [p[prefix_len:] for p in paths if p[prefix_len:]]
            combined_suffix = combine_paths(suffixes) if suffixes else None
            return path_to_join(prefix, suffix=combined_suffix)
        else:
            first, *rest = paths
            inner = combine_paths(rest)
            return f"(AND {path_to_join(first)} {inner})"

    return combine_paths(all_paths)


def merge_sexprs_with_subject(sexpr_list, roots, return_vars):
    """
    Merge multiple SExprs into one, adding SUBJECT wrappers for roots that are return variables.
    
    sexpr_list: list of SExpr strings, one per subgraph
    roots: list of roots corresponding to each SExpr
    return_vars: list of return variables for the query
    """
    if not sexpr_list:
        return ""

    wrapped_exprs = []

    for expr, root in zip(sexpr_list, roots):
        expr = expr.strip()
        # wrap with SUBJECT if root is a return variable
        if root in return_vars:
            expr = f"(SUBJECT {root} {expr})"
        
        wrapped_exprs.append(expr)

    # If only one subgraph, return it directly
    if len(wrapped_exprs) == 1:
        return wrapped_exprs[0]

    # Merge under top-level AND
    return f"(AND {' '.join(wrapped_exprs)})"

def wrap_in_select(sexpr_str, return_vars):
    """
    Wrap a final SExpr string in a SELECT clause with all return variables.
   
    Example:
      sexpr_str = "(AND ...)"
      return_vars = ["?person", "?startDate"]
    Result:
      "(SELECT ?person ?startDate (AND ...))"
    """
    if not return_vars:
        return sexpr_str  # fallback, no wrapping if empty

    vars_str = ' '.join(return_vars)
    return f"(SELECT {vars_str} {sexpr_str})"


def pretty_print_sexpr(expr: str, indent: int = 2) -> str:
    """Pretty-print an SExpr string with AND/JOIN nesting."""
    tokens = expr.replace('(', ' ( ').replace(')', ' ) ').split()
    result = []
    level = 0
    i = 0

    while i < len(tokens):
        token = tokens[i]
        if token == '(':
            # Check next token for AND/JOIN
            if i + 1 < len(tokens) and tokens[i + 1] in ('AND', 'JOIN', 'RETURN'):
                result.append(' ' * (level * indent) + '(' + tokens[i + 1])
                i += 2
                level += 1
            else:
                result.append(' ' * (level * indent) + '(')
                i += 1
        elif token == ')':
            level -= 1
            result.append(' ' * (level * indent) + ')')
            i += 1
        else:
            result.append(' ' * (level * indent) + token)
            i += 1

    return '\n'.join(result)


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
#json_file_path = "ApacheJena/test.json"

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
        input("\n\n")
        continue

    triples = extract_all_triples(ast)
    graph = build_graph(triples)

    print("Building S-Expression:")

    print("Target extraction:")
    targets = extract_select_vars(ast)
    print(targets)
    print()

    sexprs = []
    roots = []

    subgraphs = split_into_subgraphs(graph)
    for graph in subgraphs:
        print("Subgraph:")
        print(json.dumps(graph, indent=2))
        
        print("Root Selection:")
        if len(targets) > 1 or targets == []:
            roots.append(select_root_by_object_rule(graph))
        else:
            # force target to be root if only 1 target
            roots.append(targets[0])
        print(roots[len(roots) - 1])
        print()

        print("SExpr construction:")
        sexpr_schema = build_sexpr(graph, roots[len(roots) - 1])
        print(json.dumps(sexpr_schema, indent=2))
        print()

        sexpr = strip_prefixes(paths_to_sexpr(sexpr_schema, targets, roots[len(roots) - 1]))
        sexprs.append(sexpr)


    print("Sexprs list")
    print(sexprs)

    print("Final S-Expression:")
    sexpr = merge_sexprs_with_subject(sexprs, roots, targets)
    sexpr = wrap_in_select(sexpr, targets)
    print()
    print(sexpr)
    print("\n")

    print("Saved SExpr:")
    print(saved_sexpr)
    print()


    input("\n\n")
