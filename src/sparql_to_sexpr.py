import argparse
import json
import os
import yaml

from src.utils.sparql_exec import (
    normalize_gold_sparql,
    execute_sparql,
    bindings_to_rows,
)

from src.utils.kb import load_kb_module
from src.kb.base import BaseKB
from sexpr.jena_interface import fix_sparql_for_jena, detect_query_form, restore_query_form
from sexpr.jena_interface import sparql_to_algebra, algebra_to_sparql, strip_prefix_and_expand
from src.utils.run_config import apply_run_config_defaults, require

MODES  = ("jena", "sparql")
SPLITS = ("dev", "test", "train")


# ---------------------------------------------------------------------------
# File handling

def get_split_files(dataset_name: str) -> list[tuple[str, str]]:
    """
    Load raw dataset files from $(DATA_DIR)/{dataset_name}/origin/.
    Expects split files to be named {dataset_name}_{split}.json (or jsonl).
    """
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
    """
    Defines the output path of produced split files.
    """
    data_dir = os.environ.get("DATA_DIR", "data")
    name = f"{dataset_name}_{split}.{mode}.expr.json"
    return os.path.join(data_dir, dataset_name, "sexpr", name)


def build_debug_report_path(dataset_name: str, split: str, mode: str, kind: str) -> str:
    data_dir = os.environ.get("DATA_DIR", "data")
    name = f"{dataset_name}_{split}.{mode}.expr.{kind}"
    return os.path.join(data_dir, dataset_name, "sexpr", name)


def build_jsonl_scratch_path(dataset_name: str, split: str, mode: str) -> str:
    data_dir = os.environ.get("DATA_DIR", "data")
    name = f"{dataset_name}_{split}.{mode}.expr.jsonl"
    return os.path.join(data_dir, dataset_name, "sexpr", name)


def write_id_report(path: str, ids: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid in ids:
            f.write(f"{qid}\n")


# ---------------------------------------------------------------------------
# Dataset config handling

def load_config(config_path: str) -> dict:
    """
    Loads a YAML dataset config as python dict.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format: {config_path}")

    return cfg


def _apply_field_mapping(source: dict, field_map: dict) -> dict:
    """
    Applies a defined field map to an existing dictionary. All keys in the 
    target dictionary are replaced by the values of the field map (if key matches).
    """
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


def extract_flat_entries(data: list | dict, config: dict) -> list[dict]:
    """
    Apply field mapping to a dataset without nested structure.
    """
    root_key = config.get("root")

    if root_key is not None:
        if not isinstance(data, dict):
            raise ValueError(
                "'root' is only supported for dictionary-based datasets."
            )

        # Discard irrelevant meta block
        data = data.get(root_key)

    # Root is not a list
    if not isinstance(data, list):
        raise ValueError("Flat dataset must contain a list of entries.")

    field_map = config.get("fields", {})
    if not field_map:
        return data

    return [_apply_field_mapping(item, field_map) for item in data]


def extract_nested_entries(data: dict | list, config: dict) -> list[dict]:
    """
    Unpack a dataset with nested structure into a flat dataset.
    The specific behavior is controlled entirely by the dataset config.
    """
    root_key = config.get("root")
    nested_key = config.get("nested")
    parse_strategy = config.get("parse_strategy", "")
    
    if parse_strategy not in ["first", "all"]:
        raise ValueError("parse_strategy must be one of 'first' or 'all'.")

    if not nested_key:
        raise ValueError("Nested config requires 'nested'")

    inherit_map = config.get("inherit", {})
    field_map = config.get("fields", {})

    if not isinstance(field_map, dict):
        raise ValueError("'fields' must be a dictionary")

    if root_key is not None:
        if not isinstance(data, dict):
            raise ValueError("'root' requires top level dataset to be a dictionary ")
        data = data.get(root_key)

    if not isinstance(data, list):
        raise ValueError("Nested dataset must contain a list of parent entries.")
    out = []

    for parent in data:
        # Write fields of the outer map to the inner map
        inherited = {
            new_name: parent.get(old_name)
            for old_name, new_name in inherit_map.items()
        }

        children = parent.get(nested_key, [])

        if not isinstance(children, list):
            continue

        # Convert only the first parse
        if parse_strategy == "first":
            chosen = None
            if children:
                chosen = children[0]

            if chosen is not None:
                entry = dict(inherited)
                entry.update(_apply_field_mapping(chosen, field_map))
                out.append(entry)

        # Create a dataset item for every parse
        else:
            for child in children:
                entry = dict(inherited)
                entry.update(_apply_field_mapping(child, field_map))
                out.append(entry)

    return out


def load_dataset(path: str, config: dict | None = None) -> list[dict]:
    """
    Loads a raw dataset split file by first transforming it into what the pipeline expects.
    """
    # Load file
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)

    if not config:
        raise ValueError("No dataset config provided.")
    
    fmt = config.get("format", "flat")

    if fmt == "nested":
        return extract_nested_entries(data, config)

    if fmt == "flat":
        return extract_flat_entries(data, config)

    raise ValueError(
        f"Unsupported dataset format '{fmt}'. "
        "Expected 'flat' or 'nested'."
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
    kb: BaseKB,
    config: dict | None = None,
    no_mismatch_analysis: bool = False,
    no_gold_exec: bool = False,
) -> dict:

    print(f"\n{'=' * 60}")
    print(f"Split: {split} | Mode: {mode}")
    print(f"Input: {input_path}")
    if no_gold_exec:
        print((
            "[WARN] All gold execution disabled! This will break the retrieval step later on."
            "       Only use this if you intend for the dataset to be used for training only."
        ))
        
    elif no_mismatch_analysis:
        print((
            "[WARN] Mismatch analysis disabled! This will not verify whether normed SPARQL and"
            "       raw gold SPARQL of the dataset produce the same results. Use this only, if"
            "       the raw gold SPARQL queries inherently fail on the used endpoint."
            ))

    entries = load_dataset(input_path, config)
    total = len(entries)
    
    # Accumulators
    conv_failed  = 0
    conv_skipped = 0
    exec_ok = 0
    exec_failed  = 0
    raw_exec_failed = 0
    stale_count = 0 # gold execution succeeded but returned nothing
    mismatch_count = 0 # raw gold vs normed gold results differ
    failed_ids: list[str] = []
    stale_ids: list[str] = []
    mismatch_ids: list[str] = []
    
    converter = CONVERTERS.get(mode)
    endpoint_url = os.environ.get("ENDPOINT_URL")

    if not endpoint_url:
        print("[ERROR]: ENDPOINT_URL not set")
        return

    # Create scratch file for incremental write
    jsonl_path = build_jsonl_scratch_path(dataset_name, split, mode)
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    # Process entries
    for i, entry in enumerate(entries):
        sparql_query = (entry.get("sparql") or "").strip()
        qid = entry.get("id", f"entry-{i}")

        print(f"[{i+1}/{total}] {qid}", end=" ... ", flush=True)

        # Normalize gold SPARQL query
        normed, norm_err = (
            normalize_gold_sparql(sparql_query, kb.COMMON_PREFIXES)
            if sparql_query else (None, None)
        )
        entry["normed_sparql"] = normed
        if norm_err:
            entry["normed_sparql_error"] = norm_err

        raw_rows = None
        normed_rows = None

        if not no_gold_exec:
            if not no_mismatch_analysis:
                # Execute raw gold query
                if sparql_query and endpoint_url:
                    raw_result = execute_sparql(sparql_query, endpoint_url, timeout=300)
                    if raw_result is not None:
                        raw_rows = bindings_to_rows(raw_result, kb)
                        entry["gold_raw_answer"] = raw_rows
                        # Empty results -> stale
                        if not raw_rows:
                            stale_count += 1
                            stale_ids.append(qid)
                    else:
                        entry["gold_raw_exec_failed"] = True
                        raw_exec_failed += 1

            # Execute the normalized gold query
            if normed and endpoint_url:
                normed_result = execute_sparql(normed, endpoint_url, timeout=300)
                if normed_result is not None:
                    normed_rows = bindings_to_rows(normed_result, kb)
                    entry["answer"] = normed_rows
                    exec_ok += 1

                    # Use normalized SPARQL query results as stale detection if mismatch analysis is not used
                    if no_mismatch_analysis and not normed_rows:
                        stale_count += 1
                        stale_ids.append(qid)
                else:
                    entry["answer_exec_failed"] = True
                    exec_failed += 1

            # Compare raw gold vs normed gold results
            if not no_mismatch_analysis and raw_rows is not None and normed_rows is not None:
                if {tuple(r) for r in raw_rows} != {tuple(r) for r in normed_rows}:
                    entry["gold_normed_mismatch"] = True
                    mismatch_count += 1
                    mismatch_ids.append(qid)

        # Training target conversion
        if not sparql_query:
            entry["Sexpr"] = "Parsing failed"
            conv_skipped += 1
            print("[INFO] skipping item with no associated query")
        else:
            try:
                fixed = fix_sparql_for_jena(sparql_query, kb.COMMON_PREFIXES)
                # Fix "ASK" queries
                form = detect_query_form(fixed)
                entry["Sexpr"] = converter(fixed, kb.COMMON_PREFIXES, form)
                print("[INFO] Parse OK")

            except Exception as e:
                entry["Sexpr"] = "Parsing failed"
                conv_failed += 1
                failed_ids.append(qid)
                print(f"[ERROR] Parse Failed: ({e})")

        jsonl_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    jsonl_f.close()

    norm_failed = sum(1 for e in entries if e.get("normed_sparql_error"))
    conv_ok = total - conv_failed - conv_skipped

    # Print statistics
    print(f"\nConversion : {conv_ok}/{total} ok, {conv_failed} failed, {conv_skipped} skipped")
    print(f"Gold norm  : {total - norm_failed}/{total} ok, {norm_failed} failed")
    if no_gold_exec:
        print(f"Gold exec  : skipped")
    elif no_mismatch_analysis:
        print(f"Gold exec (normed) : {exec_ok}/{total} ok, {exec_failed} failed, {stale_count} empty")
    else:
        print(f"Gold exec (raw)    : {raw_exec_failed} failed, {stale_count} empty")
        print(f"Gold exec (normed) : {exec_ok}/{total} ok, {exec_failed} failed")
        print(f"Raw vs normed mismatch : {mismatch_count}")

    # Write processed split files
    out_path = build_output_path(dataset_name, split, mode)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        final_entries = [json.loads(line) for line in f if line.strip()]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_entries, f, indent=2, ensure_ascii=False)

    os.remove(jsonl_path)

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
    """
    Prints overall statistics about success rate, errors and mismatches per split.
    """
    print(f"\n{'=' * 60}")
    print("Overview (all splits)")
    print(f"{'=' * 60}")

    for r in results:
        print(f"\nSplit: {r['split']}")
        print(f"  Conversion : {r['conv_ok']}/{r['total']} ok, {r['conv_failed']} failed, {r['conv_skipped']} skipped")
        print(f"  Gold norm  : {r['total'] - r['norm_failed']}/{r['total']} ok, {r['norm_failed']} failed")
        print(f"  Empty gold results (stale dataset?) : {r['stale_count']}")
        print(f"  Raw vs normed gold result mismatch  : {r['mismatch_count']}")

        # Write IDs of problematic dataset items to seperate debug files
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
    parser.add_argument("--mode", choices=MODES, default="sparql", help="Conversion target")
    parser.add_argument(
        "--kb", 
        default="wikidata",                
        help=(
            "KB module. The KB module defines the SPARQL query used"
            "to retrieve entity and predicate labels."
        )
    )
    parser.add_argument("--config", default=None, help="Optional YAML config for dataset")
    parser.add_argument("--run_config", type=str, default=None,
                        help="Path to configs/run/<name>.yaml"
                        )
    parser.add_argument(
        "--no_mismatch_analysis",
        action="store_true",
        default=False,
        help=(
            "Skip raw (unnormalised) gold execution and the raw-vs-normed "
            "mismatch check. Use this for datasets whose gold SPARQL doesn't "
            "declare prefixes, so raw execution would fail for every entry. "
            "Staleness ('empty result') detection falls back to the normed "
            "gold query in this mode."
        ),
    )
    parser.add_argument(
        "--no_gold_exec",
        action="store_true",
        default=False,
        help=(
            "Skip gold execution entirely -- neither the raw nor the normed gold "
            "query is run against the endpoint, and no 'answer' / 'gold_raw_answer' "
            "fields are written. Use this when some gold queries return results too "
            "large to hold in memory, or time out against the endpoint. Implies "
            "--no_mismatch_analysis (nothing left to compare)."
        ),
    )

    apply_run_config_defaults(parser, section="convert", config_ref_key="dataset_config")

    args = parser.parse_args()
    require(args, "dataset")

    # Load dataset split files
    dataset = args.dataset
    splits = get_split_files(dataset)
    
    if not splits:
        print(f"No files found for '{dataset}'")
        return
    
    print(f"Found splits: {', '.join(s for s, _ in splits)}")
    
    # Load KB module
    kb_module = load_kb_module(args.kb)
    print(f"Loaded {len(kb_module.COMMON_PREFIXES)} prefixes from {args.kb}")

    # Load dataset config
    config = None
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config: {args.config}")

    # Process splits
    results = []
    for split, path in splits:
        result = process_split(
            dataset, split, path, args.mode, kb_module, config,
            no_mismatch_analysis=args.no_mismatch_analysis,
            no_gold_exec=args.no_gold_exec,
        )
        results.append(result)

    print_final_overview(results)

    print("\nDone.")


if __name__ == "__main__":
    main()