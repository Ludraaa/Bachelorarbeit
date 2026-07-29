"""
SPARQL filter and expression serialisation.

expr_to_str is the single entry-point; it handles infix operators,
unary operators, and all built-in function calls.  It intentionally does
NOT overlap with normalize.py (IRI handling) or normalize.path_to_str
(property paths).
"""

from __future__ import annotations
from typing import Union

from sexpr.normalize import normalize_iri, sexp_to_str

SExp = Union[str, list]
PrefixMap = dict[str, str]

INFIX_OPS: frozenset[str] = frozenset({
    "=", "!=", "<", ">", "<=", ">=",
    "&&", "||",
    "+", "-", "*", "/",
})


def expr_to_str(node: SExp, prefix_map: PrefixMap) -> str:
    """Render a filter/expression node as a readable SPARQL expression string."""

    # --- atoms ---
    if isinstance(node, str):
        if "^^" in node:
            val, typ = node.split("^^", 1)
            resolved = normalize_iri(typ, prefix_map) if typ.startswith("<") else typ
            return f"{val}^^{resolved}"
        return normalize_iri(node, prefix_map)

    if not isinstance(node, list) or not node:
        return str(node)

    op = node[0]

    # --- infix binary ---
    if op in INFIX_OPS and len(node) == 3:
        left  = expr_to_str(node[1], prefix_map)
        right = expr_to_str(node[2], prefix_map)
        return f"{left} {op} {right}"

    # --- unary not ---
    if op == "!" and len(node) == 2:
        return f"!{expr_to_str(node[1], prefix_map)}"

    # --- conjunction list (Jena sometimes emits this instead of nested &&) ---
    if op == "exprlist":
        return " && ".join(expr_to_str(child, prefix_map) for child in node[1:])

    # --- single-argument builtins ---
    _SINGLE_ARG = {
        "lang":      "LANG",
        "isliteral": "isLiteral",
        "bound":     "BOUND",
        "str":       "STR",
        "sample":    "SAMPLE",
    }
    if op in _SINGLE_ARG:
        return f"{_SINGLE_ARG[op]}({expr_to_str(node[1], prefix_map)})"

    # --- two-argument builtins ---
    if op == "langmatches":
        return (
            f"LANGMATCHES("
            f"{expr_to_str(node[1], prefix_map)}, "
            f"{expr_to_str(node[2], prefix_map)})"
        )

    # --- variadic builtins ---
    if op in ("regex", "concat"):
        args = ", ".join(expr_to_str(a, prefix_map) for a in node[1:])
        return f"{op.upper()}({args})"

    # --- set membership ---
    if op in ("in", "notin"):
        left   = expr_to_str(node[1], prefix_map)
        values = ", ".join(expr_to_str(v, prefix_map) for v in node[2:])
        keyword = "IN" if op == "in" else "NOT IN"
        return f"{left} {keyword} ({values})"

    # --- IRI constructor (Jena may insert a file:// base URI as first arg) ---
    if op == "iri":
        args = [
            a for a in node[1:]
            if not (isinstance(a, str) and a.startswith('"file://'))
        ]
        return f"IRI({expr_to_str(args[0], prefix_map)})"

    # --- fallback: unknown function in uppercase call style ---
    args = ", ".join(expr_to_str(a, prefix_map) for a in node[1:])
    return f"{op.upper()}({args})"