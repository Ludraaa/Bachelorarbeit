"""
Algebra AST -> Simple format transformation.

  extract_prefix_map(ast)            -> PrefixMap
  algebra_to_simple(node, ...)       -> str

Internal helpers are prefixed with underscore and not exported.

Design notes
------------
* algebra_to_simple handles the structural / query-level operators:
    prefix, project, distinct, slice, and the ASK fallback.

* _collect_blocks handles body-level operators and returns
    (required_lines, optional_lines).  It never calls algebra_to_simple
    except for the subquery case (op == "project"), making the
    mutual recursion explicit and bounded.

* _try_collapse_extend_group returns Optional[SExp] (None = no match)
  instead of using an identity sentinel, making the contract clear.
"""

from __future__ import annotations

from typing import Optional, Union

from sexpr.normalize import normalize_iri, term_to_str, path_to_str, sexp_to_str
from sexpr.expr import expr_to_str

SExp      = Union[str, list]
PrefixMap = dict[str, str]
Lines     = list[str]

# ---------------------------------------------------------------------------
# Prefix extraction
# ---------------------------------------------------------------------------

def extract_prefix_map(ast: SExp, COMMON_PREFIXES) -> PrefixMap:
    """Build a prefix map from the AST, seeded with COMMON_PREFIXES."""
    prefix_map = COMMON_PREFIXES.copy()
    if isinstance(ast, list) and ast[0] == "prefix":
        for p in ast[1]:
            prefix = p[0].rstrip(":")
            uri    = p[1][1:-1]   # strip the surrounding <>
            prefix_map[prefix] = uri
    return prefix_map


# ---------------------------------------------------------------------------
# Lang-filter stripping
# ---------------------------------------------------------------------------

def _is_lang_filter(node: SExp) -> bool:
    """Return True for  (= (lang ?var) "en")  (English lang filters only)"""
    return (
        isinstance(node, list)
        and len(node) == 3
        and node[0] == "="
        and isinstance(node[1], list)
        and node[1][0] == "lang"
        and node[2] == '"en"'
    )


def _strip_lang_filters(node: SExp) -> Optional[SExp]:
    """
    Remove English lang-equality filters from an expression tree.
    Returns None when the entire expression has been stripped.
    """
    if _is_lang_filter(node):
        return None
    if not isinstance(node, list):
        return node

    op = node[0]

    if op == "exprlist":
        cleaned = [_strip_lang_filters(c) for c in node[1:]]
        cleaned = [c for c in cleaned if c is not None]
        if not cleaned:
            return None
        return cleaned[0] if len(cleaned) == 1 else ["exprlist"] + cleaned

    if op == "&&":
        left  = _strip_lang_filters(node[1])
        right = _strip_lang_filters(node[2])
        if left  is None: return right
        if right is None: return left
        return ["&&", left, right]

    return node


# ---------------------------------------------------------------------------
# extend+group collapsing
# ---------------------------------------------------------------------------

def _try_collapse_extend_group(node: SExp) -> Optional[SExp]:
    """
    If node is a chain of (extend ...) nodes wrapping a (group ...), collapse
    them by mapping internal ?.N variables back to user-visible names.

    Returns the collapsed node on success, or None if the structure does not
    match (so callers can branch cleanly without identity comparisons).
    """
    if not (isinstance(node, list) and node[0] == "extend"):
        return None

    all_bindings: list = []
    current = node
    while isinstance(current, list) and current[0] == "extend":
        all_bindings.extend(current[1])
        current = current[2]

    if not (isinstance(current, list) and current[0] == "group"):
        return None

    reverse = {
        b[1]: b[0]
        for b in all_bindings
        if isinstance(b[1], str) and b[1].startswith("?.")
    }
    if not reverse:
        return None

    group_vars = current[1]
    agg_exprs  = current[2]
    body       = current[3]
    new_aggs   = [[reverse.get(a[0], a[0]), a[1]] for a in agg_exprs]
    return ["group", group_vars, new_aggs, body]


# ---------------------------------------------------------------------------
# Internal-variable map for ORDER BY resolution
# ---------------------------------------------------------------------------

def _get_internal_var_map(node: SExp, prefix_map: PrefixMap) -> dict[str, str]:
    """
    Walk node to find the nearest (group ...) and return a mapping of
    internal variables (?.0, ?.1, …) to their aggregate expression strings.

    Returns an empty dict when no group is found (ORDER BY will then use
    the raw variable names, which is safe).
    """
    if not isinstance(node, list):
        return {}
    if node[0] == "group":
        return {
            agg[0]: sexp_to_str(agg[1], prefix_map)
            for agg in node[2]
            if isinstance(agg[0], str) and agg[0].startswith("?.")
        }
    for child in node[1:]:
        result = _get_internal_var_map(child, prefix_map)
        if result:
            return result
    return {}


# ---------------------------------------------------------------------------
# Filter splitting (inline vs block constructs)
# ---------------------------------------------------------------------------

def _split_filter_expr(
    node: SExp,
    indent: int,
    prefix_map: PrefixMap,
) -> tuple[Lines, Optional[str]]:
    """
    Partition a filter expression into:
      block_lines  - NOT EXISTS / EXISTS clauses rendered as indented blocks
      inline_expr  - everything else as a single expression string, or None

    This keeps NOT EXISTS / EXISTS out of a FILTER(...) wrapper where they
    would be syntactically invalid.
    """
    pad = "  " * indent

    if not isinstance(node, list):
        return [], expr_to_str(node, prefix_map)

    op = node[0]

    if op == "notexists":
        req, opt = _collect_blocks(node[1], indent + 1, prefix_map)
        return [f"{pad}NOT EXISTS\n" + "\n".join(req + opt)], None

    if op == "exists":
        req, opt = _collect_blocks(node[1], indent + 1, prefix_map)
        return [f"{pad}EXISTS\n" + "\n".join(req + opt)], None

    if op in ("exprlist", "&&"):
        children = node[1:] if op == "exprlist" else [node[1], node[2]]
        all_blocks: Lines = []
        inline_parts: list[str] = []
        for child in children:
            blocks, inline = _split_filter_expr(child, indent, prefix_map)
            all_blocks.extend(blocks)
            if inline:
                inline_parts.append(inline)
        return all_blocks, (" && ".join(inline_parts) if inline_parts else None)

    return [], expr_to_str(node, prefix_map)


# ---------------------------------------------------------------------------
# Block collector
# ---------------------------------------------------------------------------

def _collect_blocks(
    node: SExp,
    indent: int,
    prefix_map: PrefixMap,
) -> tuple[Lines, Lines]:
    """
    Recursively render body-level algebra nodes into two lists:
      required  - lines that must appear before OPTIONAL / FILTER / MINUS
      optional  - OPTIONAL, FILTER, MINUS, and similar modifier blocks

    The caller is responsible for joining and indenting the final output.
    """
    pad = "  " * indent

    if not isinstance(node, list):
        return [], []

    op = node[0]

    # --- join: merge both sides ---
    if op == "join":
        req1, opt1 = _collect_blocks(node[1], indent, prefix_map)
        req2, opt2 = _collect_blocks(node[2], indent, prefix_map)
        return req1 + req2, opt1 + opt2

    # --- nested SELECT = subquery ---
    if op == "project":
        inner = algebra_to_simple(node, indent + 1, prefix_map)
        return [f"{pad}SUBQUERY\n{inner}"], []

    # --- basic graph pattern ---
    if op == "bgp":
        lines = [
            f"{pad}"
            f"{term_to_str(t[1], prefix_map)} "
            f"{term_to_str(t[2], prefix_map)} "
            f"{term_to_str(t[3], prefix_map)}"
            for t in node[1:]
        ]
        return lines, []

    # --- OPTIONAL ---
    if op == "leftjoin":
        req, opt    = _collect_blocks(node[1], indent, prefix_map)
        pattern     = node[2]
        filter_cond = node[3] if len(node) > 3 else None
        if filter_cond is not None and _is_lang_filter(filter_cond):
            filter_cond = None

        body_req, body_opt = _collect_blocks(pattern, indent + 1, prefix_map)
        optional_block = f"{pad}OPTIONAL\n" + "\n".join(body_req + body_opt)
        if filter_cond is not None:
            optional_block += (
                f"\n{'  ' * (indent + 1)}"
                f"FILTER {expr_to_str(filter_cond, prefix_map)}"
            )
        return req, opt + [optional_block]

    # --- FILTER ---
    if op == "filter":
        cleaned = _strip_lang_filters(node[1])
        if cleaned is None:
            return _collect_blocks(node[2], indent, prefix_map)

        block_lines, inline_expr = _split_filter_expr(cleaned, indent, prefix_map)
        req, opt = _collect_blocks(node[2], indent, prefix_map)
        result_opt = list(opt) + block_lines
        if inline_expr:
            result_opt.append(f"{pad}FILTER {inline_expr}")
        return req, result_opt

    # --- BIND / aggregates ---
    if op == "extend":
        collapsed = _try_collapse_extend_group(node)
        if collapsed is not None:
            return _collect_blocks(collapsed, indent, prefix_map)
        # plain BIND
        bindings = node[1]
        req, opt = _collect_blocks(node[2], indent, prefix_map)
        bind_lines = [
            f"{pad}BIND {expr_to_str(b[1], prefix_map)} AS {b[0]}"
            for b in bindings
        ]
        return bind_lines + req, opt

    # --- VALUES ---
    if op == "table":
        variables = node[1][1:]   # first element is the 'vars' tag
        rows      = node[2:]
        lines     = [f"{pad}VALUES {' '.join(variables)}"]
        for row in rows:
            binding_dict = {b[0]: sexp_to_str(b[1], prefix_map) for b in row[1:]}
            values = " ".join(binding_dict.get(var, "UNDEF") for var in variables)
            lines.append(f"{pad}  {values}")
        return lines, []

    # --- GROUP BY + aggregates ---
    if op == "group":
        group_vars = node[1]
        # Jena omits the aggregate list when there are none:
        # (group (vars) body)  vs  (group (vars) ((aggs)) body)
        if len(node) == 3:
            agg_exprs, body = [], node[2]
        else:
            agg_exprs, body = node[2], node[3]

        body_req, body_opt = _collect_blocks(body, indent, prefix_map)
        lines: Lines = []
        if group_vars:
            lines.append(f"{pad}GROUP BY {' '.join(group_vars)}")
        for agg in agg_exprs:
            var, raw_expr = agg[0], agg[1]
            if isinstance(raw_expr, list) and raw_expr[0] == "count" and len(raw_expr) == 1:
                agg_str = "COUNT(*)"
            else:
                agg_str = expr_to_str(raw_expr, prefix_map)
            if not var.startswith("?."):
                lines.append(f"{pad}{agg_str} AS {var}")
        return lines + body_req + body_opt, []

    # --- sequence (ordered join) ---
    if op == "sequence":
        req, opt = [], []
        for child in node[1:]:
            r, o = _collect_blocks(child, indent, prefix_map)
            req.extend(r)
            opt.extend(o)
        return req, opt

    # --- property path triple ---
    if op == "path":
        subject   = normalize_iri(node[1], prefix_map)
        path_expr = path_to_str(node[2], prefix_map)
        obj       = normalize_iri(node[3], prefix_map)
        return [f"{pad}{subject} {path_expr} {obj}"], []

    # --- UNION ---
    if op == "union":
        left_req,  left_opt  = _collect_blocks(node[1], indent, prefix_map)
        right_req, right_opt = _collect_blocks(node[2], indent, prefix_map)
        return (
            left_req + left_opt + [f"{pad}UNION"] + right_req + right_opt,
            [],
        )

    # --- MINUS ---
    if op == "minus":
        req, opt = _collect_blocks(node[1], indent, prefix_map)
        minus_req, minus_opt = _collect_blocks(node[2], indent + 1, prefix_map)
        minus_body = "\n".join(minus_req + minus_opt)
        return req, opt + [f"{pad}MINUS\n{minus_body}"]

    # --- unknown op: emit a labelled comment so the output is still useful ---
    return [f"{pad}# UNKNOWN_OP: {sexp_to_str(node, prefix_map)}"], []


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

_BODY_OPS: frozenset[str] = frozenset({
    "join", "leftjoin", "bgp", "filter", "extend",
    "table", "group", "sequence", "path", "union", "minus",
})


def algebra_to_simple(
    node: SExp,
    indent: int = 0,
    prefix_map: Optional[PrefixMap] = None,
) -> str:
    """
    Convert a Jena algebra AST node to its Simple representation.

    Top-level operators (prefix, project, distinct, slice) are handled here.
    Body-level content is delegated to _collect_blocks.
    """
    if prefix_map is None:
        prefix_map = {}

    pad = "  " * indent

    if isinstance(node, str):
        return pad + normalize_iri(node, prefix_map)

    op = node[0]

    # --- strip prefix wrapper ---
    if op == "prefix":
        return algebra_to_simple(node[2], indent, prefix_map)

    # --- SELECT ---
    if op == "project":
        variables = " ".join(node[1])
        child     = node[2]
        lines     = [f"{pad}SELECT {variables}"]

        # Peel ORDER BEFORE processing the body, so _collect_blocks never
        # sees an (order ...) node (which it does not handle).
        order_line: Optional[str] = None
        if isinstance(child, list) and child[0] == "order":
            internal_map = _get_internal_var_map(child[2], prefix_map)
            order_parts: list[str] = []
            for c in child[1]:
                if isinstance(c, list):
                    direction = c[0].upper()
                    raw       = internal_map.get(c[1], c[1])
                    resolved  = expr_to_str(raw, prefix_map) if isinstance(raw, list) else raw
                    order_parts.append(f"{direction} {resolved}")
                else:
                    raw      = internal_map.get(c, c)
                    resolved = expr_to_str(raw, prefix_map) if isinstance(raw, list) else raw
                    order_parts.append(resolved)
            order_line = f"{pad}ORDER {' '.join(order_parts)}"
            child = child[2]

        # Strip pure lang filters that wrap the entire body.
        while isinstance(child, list) and child[0] == "filter":
            cleaned = _strip_lang_filters(child[1])
            if cleaned is None:
                child = child[2]
            else:
                break

        # Collapse extend+group chain if present.
        collapsed = _try_collapse_extend_group(child)
        if collapsed is not None:
            child = collapsed

        req, opt = _collect_blocks(child, indent + 1, prefix_map)
        lines.extend(req + opt)
        if order_line:
            lines.append(order_line)
        return "\n".join(lines)

    # --- SELECT DISTINCT ---
    if op == "distinct":
        inner = algebra_to_simple(node[1], indent, prefix_map)
        return inner.replace("SELECT ", "SELECT DISTINCT ", 1)

    # --- LIMIT / OFFSET ---
    if op == "slice":
        offset = None if node[1] == "_" else node[1]
        limit  = None if node[2] == "_" else node[2]
        body   = algebra_to_simple(node[3], indent, prefix_map)
        suffix = ""
        if offset: suffix += f"\n{pad}OFFSET {offset}"
        if limit:  suffix += f"\n{pad}LIMIT {limit}"
        return f"{body}{suffix}"

    # --- ASK (body op at the top level = no project wrapper) ---
    if op in _BODY_OPS:
        if indent == 0:
            req, opt = _collect_blocks(node, indent + 1, prefix_map)
            return "\n".join(["ASK"] + req + opt)
        # Reached from _collect_blocks subquery path; should not happen
        req, opt = _collect_blocks(node, indent, prefix_map)
        return "\n".join(req + opt)

    raise ValueError(f"Unknown algebra op: {op!r} in node: {node}")