import argparse
import json
import os
import yaml
import importlib.util

from src.utils.sparql_exec import (
    init_uri_normaliser,
    normalise_gold_sparql,
    execute_sparql,
    bindings_to_rows,
)

from src.utils.kb import load_kb_module

from sexpr.jena_interface import fix_sparql_for_jena, detect_query_form, restore_query_form
from sexpr.jena_interface import sparql_to_algebra, algebra_to_sparql, strip_prefix_and_expand

from src.utils.run_config import apply_run_config_defaults, require


MODES  = ("jena", "sparql")
SPLITS = ("dev", "test", "train")


# ---------------------------------------------------------------------------
# file things

def get_split_files(dataset_name: str) -> list[tuple[str, str]]:
    data_dir   = os.environ.get("DATA_DIR", "data")
    origin_dir = os.path.join(data_dir, dataset_name, "origin")

    found = []

    for split in SPLITS:
        for ext in (".json", ".jsonl"):
            path = os.path.join(origin_dir, f"{dataset_name}_{split}{ext}")
            if os.path.isfile(path):
                found.append((split, path))
                break

    return found


def build_output_path(dataset_name: str, split: str, mode: str) -> str:
    data_dir = os.environ.get("DATA_DIR", "data")
    name = f"{dataset_name}_{split}.{mode}.expr.json"
    return os.path.join(data_dir, dataset_name, "sexpr", name)


def build_debug_report_path(dataset_name: str, split: str, mode: str, kind: str) -> str:
    data_dir = os.environ.get("DATA_DIR", "data")
    name = f"{dataset_name}_{split}.{mode}.expr.{kind}"
    return os.path.join(data_dir, dataset_name, "sexpr", name)


def write_id_report(path: str, ids: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid in ids:
            f.write(f"{qid}\n")


# ---------------------------------------------------------------------------
# dataset config

def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format: {config_path}")

    return cfg


def _apply_field_mapping(source: dict, field_map: dict) -> dict:
    entry: dict = {}

    for old_name, mapping in field_map.items():

        if isinstance(mapping, str):
            entry[mapping] = source.get(old_name)

        elif isinstance(mapping, dict):
            new_name = mapping.get("name")
            extract_key = mapping.get("extract")
            raw = source.get(old_name)

            if new_name is None:
                raise ValueError(f"Field mapping for '{old_name}' is missing 'name'")

            if extract_key and isinstance(raw, list):
                entry[new_name] = [
                    item.get(extract_key)
                    for item in raw
                    if isinstance(item, dict)
                ]
            else:
                entry[new_name] = raw

        else:
            raise ValueError(
                f"Invalid mapping for '{old_name}': must be a string or dict"
            )

    return entry


def extract_flat_entries(data: list, config: dict) -> list[dict]:
    """Apply field mapping to a plain list dataset."""
    field_map = config.get("fields", {})
    if not field_map:
        return data
    return [_apply_field_mapping(item, field_map) for item in data]


def extract_nested_entries(data: dict, config: dict) -> list[dict]:
    """Unpack a nested dataset and apply field mapping to each child entry."""
    root_key = config.get("root")
    nested_key = config.get("nested")
    parse_strategy = config.get("parse_strategy", "all")  # "all" | "first_success"

    if not root_key or not nested_key:
        raise ValueError("Nested config requires 'root' and 'nested'")

    inherit_map = config.get("inherit", {})
    field_map = config.get("fields", {})

    if not isinstance(field_map, dict):
        raise ValueError("'fields' must be a dictionary")

    parents = data.get(root_key)

    if not isinstance(parents, list):
        raise ValueError(f"'{root_key}' must contain a list")

    out = []

    for parent in parents:
        inherited = {
            new_name: parent.get(old_name)
            for old_name, new_name in inherit_map.items()
        }

        children = parent.get(nested_key, [])

        if not isinstance(children, list):
            continue

        if parse_strategy == "first_success":
            chosen = None
            for child in children:
                if child.get("Answers"):
                    chosen = child
                    break

            if chosen is None and children:
                chosen = children[0]

            if chosen is not None:
                entry = dict(inherited)
                entry.update(_apply_field_mapping(chosen, field_map))
                out.append(entry)

        else:  # all
            for child in children:
                entry = dict(inherited)
                entry.update(_apply_field_mapping(child, field_map))
                out.append(entry)

    return out


def load_dataset(path: str, config: dict | None = None) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)

    if config:
        fmt = config.get("format")

        if fmt == "nested":
            return extract_nested_entries(data, config)

        if isinstance(data, list):
            return extract_flat_entries(data, config)

    if isinstance(data, list):
        return data

    raise ValueError(
        "Unsupported dataset structure. "
        "Use a nested config or provide a list dataset."
    )


# ---------------------------------------------------------------------------
# conversion

def convert_raw_jena(sparql: str, common_prefixes, form) -> str:
    algebra = sparql_to_algebra(sparql).strip()
    return strip_prefix_and_expand(algebra, common_prefixes)


def convert_expanded_sparql(sparql: str, common_prefixes, form) -> str:
    algebra = sparql_to_algebra(sparql)
    no_prefix = strip_prefix_and_expand(algebra, common_prefixes)
    sparql_out = algebra_to_sparql(no_prefix).strip()
    return restore_query_form(form, sparql_out)


CONVERTERS = {
    "jena": convert_raw_jena,
    "sparql": convert_expanded_sparql,
}


# ---------------------------------------------------------------------------
# process

def process_split(
    dataset_name: str,
    split: str,
    input_path: str,
    mode: str,
    common_prefixes: list,
    config: dict | None = None,
) -> dict:

    print(f"\n{'=' * 60}")
    print(f"Split: {split} | Mode: {mode}")
    print(f"Input: {input_path}")

    entries = load_dataset(input_path, config)
    total = len(entries)
    conv_failed  = 0
    conv_skipped = 0
    exec_ok = 0          # normed gold execution ok
    exec_failed  = 0     # normed gold execution failed
    raw_exec_failed = 0  # raw (unnormalised) gold execution failed
    stale_count = 0      # raw gold execution succeeded but returned nothing
    mismatch_count = 0   # raw gold vs normed gold results differ
    failed_ids: list[str] = []
    stale_ids: list[str] = []
    mismatch_ids: list[str] = []
    converter = CONVERTERS.get(mode)
    endpoint_url = os.environ.get("ENDPOINT_URL")

    if not endpoint_url:
        print("Warning: ENDPOINT_URL not set")

    for i, entry in enumerate(entries):
        sparql_query = (entry.get("sparql") or "").strip()
        qid = entry.get("id", f"entry-{i}")

        print(f"[{i+1}/{total}] {qid}", end=" ... ", flush=True)

        # gold SPARQL normalisation
        normed, norm_err = (
            normalise_gold_sparql(sparql_query, common_prefixes)
            if sparql_query else (None, None)
        )
        entry["normed_sparql"] = normed
        if norm_err:
            entry["normed_sparql_error"] = norm_err

        raw_rows = None
        normed_rows = None

        # execute the untouched, as-given gold query
        if sparql_query and endpoint_url:
            raw_result = execute_sparql(sparql_query, endpoint_url)
            if raw_result is not None:
                raw_rows = bindings_to_rows(raw_result)
                entry["gold_raw_answer"] = raw_rows
                if not raw_rows:
                    stale_count += 1
                    stale_ids.append(qid)
            else:
                entry["gold_raw_exec_failed"] = True
                raw_exec_failed += 1

        # execute the normalised gold query (this is what downstream scoring uses)
        if normed and endpoint_url:
            normed_result = execute_sparql(normed, endpoint_url)
            if normed_result is not None:
                normed_rows = bindings_to_rows(normed_result)
                entry["answer"] = normed_rows
                exec_ok += 1
            else:
                entry["answer_exec_failed"] = True
                exec_failed += 1

        # compare raw vs normed gold execution -- catches normalisation bugs
        if raw_rows is not None and normed_rows is not None:
            if {tuple(r) for r in raw_rows} != {tuple(r) for r in normed_rows}:
                entry["gold_normed_mismatch"] = True
                mismatch_count += 1
                mismatch_ids.append(qid)

        # s-expression conversion
        if not sparql_query:
            entry["Sexpr"] = "Parsing failed"
            conv_skipped += 1
            print("SKIPPED")
            continue

        try:
            fixed = fix_sparql_for_jena(sparql_query, common_prefixes)
            form = detect_query_form(fixed)
            entry["Sexpr"] = converter(fixed, common_prefixes, form)
            print("OK")

        except Exception as e:
            entry["Sexpr"] = "Parsing failed"
            conv_failed += 1
            failed_ids.append(qid)
            print(f"FAILED ({e})")

    norm_failed = sum(1 for e in entries if e.get("normed_sparql_error"))
    conv_ok = total - conv_failed - conv_skipped

    print(f"\nConversion : {conv_ok}/{total} ok, {conv_failed} failed, {conv_skipped} skipped")
    print(f"Gold norm  : {total - norm_failed}/{total} ok, {norm_failed} failed")
    if endpoint_url:
        print(f"Gold exec (raw)    : {raw_exec_failed} failed, {stale_count} empty")
        print(f"Gold exec (normed) : {exec_ok}/{total} ok, {exec_failed} failed")
        print(f"Raw vs normed mismatch : {mismatch_count}")

    out_path = build_output_path(dataset_name, split, mode)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_path}")

    return {
        "dataset": dataset_name,
        "split": split,
        "mode": mode,
        "total": total,
        "conv_ok": conv_ok,
        "conv_failed": conv_failed,
        "conv_skipped": conv_skipped,
        "norm_failed": norm_failed,
        "exec_ok": exec_ok,
        "exec_failed": exec_failed,
        "raw_exec_failed": raw_exec_failed,
        "stale_count": stale_count,
        "mismatch_count": mismatch_count,
        "failed_ids": failed_ids,
        "stale_ids": stale_ids,
        "mismatch_ids": mismatch_ids,
    }


def print_final_overview(results: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print("Overview (all splits)")
    print(f"{'=' * 60}")

    for r in results:
        print(f"\nSplit: {r['split']}")
        print(f"  Conversion : {r['conv_ok']}/{r['total']} ok, {r['conv_failed']} failed, {r['conv_skipped']} skipped")
        print(f"  Gold norm  : {r['total'] - r['norm_failed']}/{r['total']} ok, {r['norm_failed']} failed")
        print(f"  Empty gold results (stale dataset?) : {r['stale_count']}")
        print(f"  Raw vs normed gold result mismatch  : {r['mismatch_count']}")

        for kind, ids in (
            ("failed", r["failed_ids"]),
            ("stale", r["stale_ids"]),
            ("mismatch", r["mismatch_ids"]),
        ):
            if ids:
                path = build_debug_report_path(r["dataset"], r["split"], r["mode"], kind)
                write_id_report(path, ids)
                print(f"  -> wrote {len(ids)} id(s) to {path}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SPARQL queries in dataset splits to target representation."
    )
    parser.add_argument("--dataset", default=None, help="Dataset name")
    parser.add_argument("--mode", choices=MODES, default="sparql", help="Conversion mode")
    parser.add_argument("--kb", default="wikidata", help="KB module")
    parser.add_argument("--config", default=None, help="Optional YAML config for dataset")
    parser.add_argument("--run_config", type=str, default=None,
                        help="Path to configs/run/<name>.yaml"
                        )

    apply_run_config_defaults(parser, section="convert", config_ref_key="dataset_config")

    args = parser.parse_args()
    require(args, "dataset")

    dataset = args.dataset
    splits = get_split_files(dataset)
    prefixes = []

    if args.kb:
        kb_module = load_kb_module(args.kb)
        prefixes = kb_module.COMMON_PREFIXES
        init_uri_normaliser(kb_module)
        print(f"Loaded {len(prefixes)} prefixes from {args.kb}")

    config = None

    if args.config:
        config = load_config(args.config)
        print(f"Loaded config: {args.config}")

    if not splits:
        print(f"No files found for '{dataset}'")
        return

    print(f"Found splits: {', '.join(s for s, _ in splits)}")

    results = []
    for split, path in splits:
        result = process_split(dataset, split, path, args.mode, prefixes, config)
        results.append(result)

    print_final_overview(results)

    print("\nDone.")


if __name__ == "__main__":
    main()