#!/usr/bin/env python3
"""
fuse_predictions.py

Fuses a ChatKBQA-style predictions file (JSONL: {label, predict})
with the WebQSP dataset file (JSON array) into a single JSON file.

Usage:
    python fuse_predictions.py \
        --dataset  webqsp_test.json \
        --preds    predictions.jsonl \
        --output   fused.json \
        --kb       freebase
"""

import json
import re
import argparse
from pathlib import Path

from src.utils.sparql_exec import normalise_gold_sparql
from src.utils.kb import load_kb_module


FB_NS = "http://rdf.freebase.com/ns/"


# ---------------------------------------------------------------------------
# Transform compact sexpr  (JOIN (R location.country.languages_spoken) m.03_r3)
# → adds fb: to MIDs, fbp: to dotted relation paths
# ---------------------------------------------------------------------------
def transform_compact_sexpr(sexpr: str) -> str:
    if not sexpr or sexpr == "null":
        return sexpr

    def replace_token(m: re.Match) -> str:
        token = m.group(0)
        if token.startswith('<'):
            return token
        return f'<http://rdf.freebase.com/ns/{token}>'

    return re.sub(
        r'<[^>]+>|m\.[a-z0-9_]+|[a-z][a-z_]*(?:\.[a-z_]+){1,}',
        replace_token,
        sexpr
    )


# ---------------------------------------------------------------------------
# Transform normed/labelled sexpr  ( JOIN ( R [ a , b , c d ] ) [ Jamaica ] )
# → ( JOIN (R fbp:a.b.c_d) fb:Jamaica )
# ---------------------------------------------------------------------------
def transform_normed_sexpr(sexpr: str) -> str:
    if not sexpr or sexpr == "null":
        return sexpr

    def replace_relation(m: re.Match) -> str:
        content = m.group(1).replace('，', ',')
        parts = [p.strip().replace(' ', '_') for p in content.split(',')]
        return f"(R fbp:{'.'.join(parts)})"

    result = re.sub(
        r'\(\s*R\s*\[\s*([^\]]+?)\s*\]\s*\)',
        replace_relation,
        sexpr
    )

    def replace_bracket(m: re.Match) -> str:
        content = m.group(1).strip().replace('，', ',')
        if ' , ' in content:
            parts = [p.strip().replace(' ', '_') for p in content.split(',')]
            return f'fbp:{".".join(parts)}'
        else:
            return f'fb:{content.replace(" ", "_")}'

    result = re.sub(
        r'\[\s*([^\]]+?)\s*\]',
        replace_bracket,
        result
    )

    return result


# ---------------------------------------------------------------------------
# Map expansion — short keys → full Freebase URLs
# ---------------------------------------------------------------------------
def expand_entity_map(entity_map: dict) -> dict:
    """{ "m.03_r3": "Jamaica" } → { "http://.../m.03_r3": "Jamaica" }"""
    return {f"{FB_NS}{mid}": label for mid, label in entity_map.items()}


def expand_relation_map(relation_map: dict) -> dict:
    """
    { "location.country.languages_spoken": "..." }
    → { "http://.../location.country.languages_spoken": "location.country.languages_spoken" }
    """
    return {f"{FB_NS}{rel}": rel for rel in relation_map.keys()}


def expand_type_map(type_map: dict) -> dict:
    """{ "m.0hzjlmp": "UK constituent country" } → { "http://.../m.0hzjlmp": "UK constituent country" }"""
    return {f"{FB_NS}{mid}": label for mid, label in type_map.items()}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_predictions(path: Path) -> list[dict]:
    preds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            preds.append(json.loads(line))
    return preds


def build_label_index(preds: list[dict]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for p in preds:
        index.setdefault(p["label"], []).append(p)
    return index


# ---------------------------------------------------------------------------
# Core fusion
# ---------------------------------------------------------------------------
def fuse(
    dataset: list[dict],
    preds: list[dict],
    common_prefixes,
) -> tuple[list[dict], dict[str, str]]:
    """
    Returns (fused_items, global_type_map).

    global_type_map accumulates every entry's gold_type_map (expanded to
    full URIs) across all items, without duplicates.  It is empty when the
    input dataset items carry no gold_type_map field.
    """
    label_index = build_label_index(preds)
    fused: list[dict] = []
    global_type_map: dict[str, str] = {}
    unmatched = 0

    for i, item in enumerate(dataset):
        gold_sexpr = item.get("normed_sexpr", "")

        pred_entry = None
        if gold_sexpr in label_index and label_index[gold_sexpr]:
            pred_entry = label_index[gold_sexpr].pop(0)
        elif i < len(preds):
            pred_entry = preds[i]
            if pred_entry["label"] != gold_sexpr:
                print(
                    f"[WARN] item {i} ({item.get('ID', '?')}): "
                    f"normed_sexpr != label\n"
                    f"  dataset : {gold_sexpr!r}\n"
                    f"  preds   : {pred_entry['label']!r}"
                )
            unmatched += 1

        raw_preds = pred_entry.get("predict", []) if pred_entry else []

        sparql_query = item.get("sparql", "")
        normed, _ = (
            normalise_gold_sparql(sparql_query, common_prefixes)
            if sparql_query else (None, None)
        )

        answer_list = item.get("answer", [])
        answers = [[answer] for answer in answer_list]

        # Expand and accumulate the type map (absent on non-Freebase KBs or
        # older dataset files that predate insert_labels type support).
        raw_type_map = item.get("gold_type_map", {})
        expanded_type_map = expand_type_map(raw_type_map) if raw_type_map else {}
        global_type_map.update(expanded_type_map)   # dict.update deduplicates by key

        fused_item = {
            **{k: v for k, v in item.items()
               if k not in (
                   "normed_sexpr", "gold_entity_map", "gold_relation_map",
                   "gold_type_map", "answer", "comp_type",
               )},
            "gold_entity_map":   expand_entity_map(item.get("gold_entity_map", {})),
            "gold_relation_map": expand_relation_map(item.get("gold_relation_map", {})),
            "gold_type_map":     expanded_type_map,
            "normed_sparql":     normed if normed else "",
            "sexpr":             transform_compact_sexpr(item.get("sexpr", "")),
            "sexpr_with_labels": transform_normed_sexpr(gold_sexpr),
            "debug_old_pred":    raw_preds,
            "predict":           [transform_normed_sexpr(p) for p in raw_preds],
            "answer":            answers,
        }
        fused.append(fused_item)

    if unmatched:
        print(f"[INFO] {unmatched} item(s) fell back to positional matching")
    return fused, global_type_map


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  required=True, type=Path)
    parser.add_argument("--preds",    required=True, type=Path)
    parser.add_argument("--output",   required=True, type=Path)
    parser.add_argument("--kb",       required=True)
    parser.add_argument(
        "--type-map", type=Path, default=None,
        help=(
            "Where to write the global type label map JSON. "
            "Defaults to <output-stem>_type_label_map.json alongside --output. "
            "Pass an empty string to suppress writing."
        ),
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    preds   = load_predictions(args.preds)
    kb      = load_kb_module(args.kb)
    prefixes = kb.COMMON_PREFIXES

    print(f"Loaded {len(dataset)} dataset items, {len(preds)} prediction entries")

    fused, global_type_map = fuse(dataset, preds, prefixes)

    args.output.write_text(
        json.dumps({"items": fused}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(fused)} fused items → {args.output}")

    # ------------------------------------------------------------------
    # Write the global type label map when the KB produced one.
    # Resolve output path: explicit arg > default sibling > suppress.

    if global_type_map:
        if args.type_map is None:
            type_map_path = args.output.with_name(
                args.output.stem + "_type_label_map.json"
            )
        elif str(args.type_map) == "":
            type_map_path = None
        else:
            type_map_path = args.type_map

        if type_map_path is not None:
            type_map_path.parent.mkdir(parents=True, exist_ok=True)
            type_map_path.write_text(
                json.dumps(global_type_map, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(
                f"Wrote global type label map "
                f"({len(global_type_map)} entries) → {type_map_path}"
            )
    else:
        print("No gold_type_map entries found — type label map not written "
              "(expected for non-Freebase KBs)")


if __name__ == "__main__":
    main()