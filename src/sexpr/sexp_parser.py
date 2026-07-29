"""
S-expression parser for Jena algebra output.

The grammar handled here is a subset of standard s-expressions:
  - Atoms: variables (?x), IRIs (<uri>), prefixed names (wdt:P31),
           plain strings ("..."), typed literals ("..."^^type)
  - Lists: ( child* ) or [ child* ]
"""

import re
from typing import Union

# A parsed s-expression is either a bare atom or a (possibly nested) list of atoms.
SExp = Union[str, list]


def tokenize_sexp(s: str) -> list[str]:
    return re.findall(
        r'\(|\)|\[|\]'              # delimiters
        r'|"[^"]*"\^\^[^\s()\[\]]+'  # typed literals: "val"^^type
        r'|"[^"]*"'                  # plain string literals
        r'|[^\s()\[\]]+',            # everything else (atoms)
        s,
    )


def _parse_tokens(tokens: list[str], i: int) -> tuple[SExp, int]:
    """Recursive descent parser. Returns (parsed_node, next_index)."""
    if i >= len(tokens):
        raise ValueError(f"Unexpected end of token stream at position {i}")

    tok = tokens[i]

    if tok in ("(", "["):
        close = ")" if tok == "(" else "]"
        i += 1
        children: list[SExp] = []
        while True:
            if i >= len(tokens):
                raise ValueError(
                    f"Unclosed '{tok}': reached end of token stream"
                )
            if tokens[i] == close:
                return children, i + 1
            child, i = _parse_tokens(tokens, i)
            children.append(child)

    elif tok in (")", "]"):
        raise ValueError(f"Unexpected closing '{tok}' at position {i}")

    else:
        return tok, i + 1


def parse_sexp(s: str) -> SExp:
    """Parse a full s-expression string into a nested list / atom tree."""
    tokens = tokenize_sexp(s.strip())
    if not tokens:
        raise ValueError("Empty s-expression")
    result, consumed = _parse_tokens(tokens, 0)
    if consumed != len(tokens):
        raise ValueError(
            f"Trailing tokens after position {consumed}: {tokens[consumed:]}"
        )
    return result