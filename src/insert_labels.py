import json
import re
import sys
import argparse
import os
from pathlib import Path
import requests
from src.utils.retry import call_with_retry
from src.utils.kb import load_kb_module
from src.utils.run_config import apply_run_config_defaults, require


ENDPOINT_URL = os.getenv("ENDPOINT_URL")
BATCH_SIZE = 50

SPLITS = ("dev", "test", "train")

_SPARQL_HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "insert_labels/1.0",
}

# REGEX to match any IRI
URI_PATTERN = re.compile(r"<(https?://[^>]+)>")

# ---------------------------------------------------------------------------
# Debug handling

_debug = False

def debug(*args, **kwargs):
    if _debug:
        print("[DEBUG]", *args, **kwargs)


# ---------------------------------------------------------------------------
# Failure tracking

# URI -> failure info
_label_failures: dict[str, dict] = {}
_relation_failures: dict[str, dict] = {}


def log_label_failure(uri: str, reason: str, details: str = ""):
    """
    Tracks label resolution failures for final reporting.
    """
    if uri not in _label_failures:
        _label_failures[uri] = {
            "reason": reason,
            "details": details,
            "count": 1,
            "examples": []
        }
    else:
        _label_failures[uri]["count"] += 1


def log_relation_failure(uri: str, reason: str, details: str = ""):
    """
    Tracks relation label resolution failures for final reporting.
    """
    if uri not in _relation_failures:
        _relation_failures[uri] = {
            "reason": reason,
            "details": details,
            "count": 1,
            "examples": []
        }
    else:
        _relation_failures[uri]["count"] += 1


def print_failure_report():
    """
    Prints detailed failure report at the end.
    """
    if not _debug:
        return
    
    print("\n" + "-"*80)
    print("Label Resolution failure report")
    
    # Entity failures
    print(f"\n--- Entity Label Failures: {len(_label_failures)} unique IRIs ---")
    if not _label_failures:
        print("    None - all entities resolved successfully!")
    else:
        # Group by reason
        by_reason: dict[str, list] = {}
        for uri, info in _label_failures.items():
            by_reason.setdefault(info["reason"], []).append((uri, info))
        
        for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
            print(f"\n    Reason: {reason} ({len(items)} URIs)")
            # Show top 10 examples
            for uri, info in sorted(items, key=lambda x: -x[1]["count"])[:10]:
                print(f"        - {uri} (occurred {info['count']} times)")
                if info["details"]:
                    print(f"            Details: {info['details']}")
                if info["examples"]:
                    example_ids = info["examples"][:5]
                    print(f"            Example entries: {', '.join(example_ids)}")
    
    # Relation failures
    print(f"\n--- Relation Label Failures: {len(_relation_failures)} unique IRIs ---")
    if not _relation_failures:
        print("    None - all relations resolved successfully!")
    else:
        by_reason = {}
        for uri, info in _relation_failures.items():
            by_reason.setdefault(info["reason"], []).append((uri, info))
        
        for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
            print(f"\n    Reason: {reason} ({len(items)} URIs)")
            for uri, info in sorted(items, key=lambda x: -x[1]["count"])[:10]:
                print(f"        - {uri} (occurred {info['count']} times)")
                if info["details"]:
                    print(f"            Details: {info['details']}")
                if info["examples"]:
                    example_ids = info["examples"][:5]
                    print(f"            Example entries: {', '.join(example_ids)}")
    
    print("\n" + "-"*80)


# ---------------------------------------------------------------------------
# Endpoint resolution


def resolve_endpoint(kb) -> str:
    return getattr(kb, "LABEL_ENDPOINT_URL", None) or ENDPOINT_URL


# ---------------------------------------------------------------------------
# File handling

def discover_paths(dataset: str) -> dict[str, dict[str, dict]]:
    """
    Dynamically discovers all available input files for the specified dataset.
    Processes every split of any training format files that exist. 
    """
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    base = data_dir / dataset
    input_dir = base / "sexpr"

    result: dict[str, dict[str, dict]] = {}

    for split in SPLITS:
        grouped: dict[str, list[Path]] = {}
        # Search files of any split and mode
        for f in input_dir.glob(f"{dataset}_{split}*.expr.json"):
            parts = f.stem.split(".")
            if len(parts) < 3:
                continue
            mode = parts[-2]
            grouped.setdefault(mode, []).append(f)

        if not grouped:
            continue

        result[split] = {
            mode: {
                "inputs": sorted(paths),
                "merged": base / "generation" / "merged" / f"{dataset}_{split}.{mode}.json",
                "label_cache": base / "cache" / "labels.json",
            }
            for mode, paths in grouped.items()
        }

    return result


def load_cache(path: Path) -> dict:
    """
    Loads the dataset-specific label or type cache.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(cache: dict, path: Path) -> None:
    """
    Writes the dataset-specific label or type cache.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Label fetching


def fetch_labels(uris: list[str], cache: dict, kb) -> dict:
    """
    For a given list of IRIs, batch fetches their labels.
    The query used to fetch these labels is specified by
    the KB module. Results are cached to prevent querying
    the potentially small set of entities and predicate across dataset.
    
    """
    uri_map: dict[str, list[str]] = {}
    
    n_cached = 0
    n_null = 0
    n_queued = 0
    
    for uri in uris:
        if uri in cache:
            if cache[uri] is not None:
                n_cached += 1
            else:
                n_null += 1
                # Track cached failures
                if "/entity/Q" in uri or "/entity/P" in uri:
                    log_label_failure(uri, "cached_null", "Previously cached as no-label")
                elif "/prop/" in uri:
                    log_relation_failure(uri, "cached_null", "Previously cached as no-label")
            continue
        
        # Get label-bearing IRI
        norm = kb.normalize(uri)
        uri_map.setdefault(norm, []).append(uri)
        n_queued += 1
    
    debug(f"   fetch_labels: {n_cached} already cached with labels, {n_null} no-label, {n_queued} queued for SPARQL")
    
    # Nothing to fetch
    missing = list(uri_map)
    if not missing:
        debug("  nothing to fetch")
        return cache
    
    endpoint = resolve_endpoint(kb)
    
    total_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[INFO]     fetching {len(missing)} labels in {total_batches} batches")
    debug(f"   endpoint: {endpoint}")
    
    # Batch fetch labels
    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i : i + BATCH_SIZE]
        batch_id = i // BATCH_SIZE + 1
        
        values = " ".join(f"<{u}>" for u in batch)
        query = kb.LABEL_QUERY.format(values=values, language=kb.LANGUAGE)
        
        debug(f"   batch {batch_id}/{total_batches}: {len(batch)} IRIs")
        
        try:
            resp = call_with_retry(
                requests.post,
                endpoint,
                data={"query": query},
                headers=_SPARQL_HEADERS,
                retries=5,
                base_delay=1.0,
                backoff=2.0,
                exceptions=(requests.RequestException,),
            )
            
            # Parse label query results
            payload = resp.json()
            bindings = payload["results"]["bindings"]
            found = kb.parse_label_results(bindings)
            
            for norm_uri, label in found.items():
                for orig in uri_map.get(norm_uri, [norm_uri]):
                    cache[orig] = label
            
            missed = [norm for norm in batch if norm not in found]
            for norm in missed:
                for orig in uri_map.get(norm, []):
                    cache[orig] = None
                    # Track SPARQL misses
                    if "/entity/Q" in orig or "/entity/P" in orig:
                        log_label_failure(orig, "sparql_no_label", f"SPARQL returned no label for normalized URI {norm}")
                    elif "/prop/" in orig:
                        log_relation_failure(orig, "sparql_no_label", f"SPARQL returned no label for normalized URI {norm}")
        
        except Exception as e:
            print(f"[WARN]    batch {batch_id}/{total_batches} failed: {e}")
            # Track batch failures
            for norm in batch:
                for orig in uri_map.get(norm, []):
                    cache[orig] = None
                    if "/entity/Q" in orig or "/entity/P" in orig:
                        log_label_failure(orig, "sparql_error", f"Batch failed: {e}")
                    elif "/prop/" in orig:
                        log_relation_failure(orig, "sparql_error", f"Batch failed: {e}")
            
            if _debug:
                import traceback
                traceback.print_exc()
    
    return cache


# ---------------------------------------------------------------------------
# Type fetching


def fetch_types(uris: list[str], types_cache: dict[str, bool], kb) -> dict[str, bool]:
    """
    Given a list of IRIs, this function executes the type query against the endpoint
    to determine whether they are a type/class in the context of the KB. Results are
    cached because many entitites appear multiple times in a dataset.
    """
    missing = [u for u in uris if u not in types_cache]
    # Nothing to fetch
    if not missing:
        return types_cache

    endpoint = resolve_endpoint(kb)

    total_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[INFO]     fetching type membership for {len(missing)} IRIs in {total_batches} batches")
    debug(f"   endpoint: {endpoint}")

    # Batch fetch type membership
    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i : i + BATCH_SIZE]
        batch_id = i // BATCH_SIZE + 1

        values = " ".join(f"<{u}>" for u in batch)
        query = kb.TYPES_QUERY.format(values=values)

        debug(f"   types batch {batch_id}/{total_batches}: {len(batch)} IRIs")

        try:
            resp = call_with_retry(
                requests.post,
                endpoint,
                data={"query": query},
                headers=_SPARQL_HEADERS,
                retries=5,
                base_delay=1.0,
                backoff=2.0,
                exceptions=(requests.RequestException,),
            )
            # Parse type membership query results
            bindings = resp.json()["results"]["bindings"]
            type_uris = kb.parse_type_results(bindings)
            debug(f"   parse_type_results -> {len(type_uris)} type IRIs")

            for u in batch:
                types_cache[u] = (u in type_uris)

        except Exception as e:
            print(f"[WARN]     types batch {batch_id}/{total_batches} failed: {e}")
            if _debug:
                import traceback
                traceback.print_exc()
            # Mark as False so failing queries are not retried forever
            for u in batch:
                if u not in types_cache:
                    types_cache[u] = False

    return types_cache


def apply_labels(sexpr: str, cache: dict, kb, entry_id: str = "") -> str:
    """
    Converts the training format representation into its label-enriched form by
    replacing IRIs with {prefix}:{label}. This is controlled by the KB module's
    format_label function.
    """
    formatter = getattr(kb, "format_label", None)

    def repl(match: re.Match) -> str:
        uri = match.group(1)
        label = cache.get(uri)

        out = formatter(uri, label)
        debug(f"     apply_labels: {uri} -> {out!r} (via formatter, label={label!r})")
        return out

    return URI_PATTERN.sub(repl, sexpr)


# ---------------------------------------------------------------------------
# Processing


def action_merge_all(
    dataset_name: str,
    split: str,
    dataset: list,
    cache: dict,
    kb,
    paths: dict,
    types_cache: dict[str, bool] | None = None,
) -> dict[str, str]:
    """
    Transforms a single input split file into the label-enriched version.
    Queries and substitutes labels into the training format, records the transformations
    into gold entity and predicate maps. Also checks type membership for every entity
    and writes to both the items gold type map, as well as the global type map accordingly.
    """

    # Determines whether the provided KB module uses type map at all
    has_types = (
        types_cache is not None
        and hasattr(kb, "TYPES_QUERY")
        and hasattr(kb, "parse_type_results")
    )

    format_rel = getattr(kb, "format_relation_label", None)

    print("[INFO]     Collecting IRIs...")
    all_uris: set[str] = set()

    # Extract all IRIs from training format file
    for entry in dataset:
        s = entry.get("Sexpr", "")
        if s and s != "Parsing failed":
            all_uris.update(URI_PATTERN.findall(s))

    missing_labels = [u for u in all_uris if u not in cache]
    print(f"[INFO]     {len(all_uris)} IRIs ({len(missing_labels)} new)")
    debug(f"   Sample URIs from sexprs: {sorted(all_uris)[:5]}")

    # Fetch missing labels
    if missing_labels:
        fetch_labels(list(all_uris), cache, kb)
        save_cache(cache, paths["label_cache"])

        n_resolved = sum(1 for u in all_uris if cache.get(u))
        n_none     = sum(1 for u in all_uris if u in cache and cache[u] is None)
        debug(f"   After fetch: {n_resolved} resolved, {n_none} null, {len(all_uris) - n_resolved - n_none} still missing")

    # Fetch type membership
    if has_types:
        # Collect all entity IRIs
        all_entity_uris = list({
            uri
            for entry in dataset
            for uri in kb.extract_entities(entry.get("Sexpr", ""))
            if entry.get("Sexpr", "") not in ("", "Parsing failed")
        })
        missing_types = [u for u in all_entity_uris if u not in types_cache]
        if missing_types:
            fetch_types(all_entity_uris, types_cache, kb)
        else:
            debug(f"   Type membership already cached for all {len(all_entity_uris)} entity IRIs")

    print("[INFO]     Building output...")
    merged = []
    global_type_map: dict[str, str] = {}
    n_empty = n_ok = 0

    for entry in dataset:
        sexpr = entry.get("Sexpr", "")

        record = {
            "ID": entry.get("id", ""),
            "question": entry.get("question", ""),
            "answer": entry.get("answer", []),
            "sexpr": sexpr,
            "sparql": entry.get("sparql", ""),
            "normed_sparql": entry.get("normed_sparql", ""),
        }

        if sexpr == "Parsing failed":
            record.update({
                "sexpr_with_labels": "",
                "gold_entity_map": {},
                "gold_relation_map": {},
            })
            if has_types:
                record["gold_type_map"] = {}
            merged.append(record)
            n_empty += 1
            continue

        entities = kb.extract_entities(sexpr)
        relations = kb.extract_relations(sexpr)

        debug(f"   Entry {entry.get('id', '?')}: {len(entities)} entities, {len(relations)} relations")

        ent_map = {u: cache[u] for u in entities if cache.get(u)}
        rel_map = {
            u: (format_rel(u, cache[u]) if format_rel else cache[u])
            for u in relations if u in cache
        }

        if _debug:
            unresolved_ents = [u for u in entities if not cache.get(u)]
            unresolved_rels = [u for u in relations if not cache.get(u)]
            if unresolved_ents or unresolved_rels:
                debug(f"        unresolved entities: {unresolved_ents}")
                debug(f"        unresolved relations: {unresolved_rels}")
                # Add example context to failure tracking
                entry_id = entry.get("id", "?")
                for u in unresolved_ents:
                    if u in _label_failures:
                        if entry_id not in _label_failures[u]["examples"]:
                            _label_failures[u]["examples"].append(entry_id)
                for u in unresolved_rels:
                    if u in _relation_failures:
                        if entry_id not in _relation_failures[u]["examples"]:
                            _relation_failures[u]["examples"].append(entry_id)

        record.update({
            "sexpr_with_labels": apply_labels(sexpr, cache, kb, entry.get("id", "")),
            "gold_entity_map": ent_map,
            "gold_relation_map": rel_map,
        })

        if has_types:
            type_map = {
                u: label
                for u, label in ent_map.items()
                if types_cache.get(u)
            }
            record["gold_type_map"] = type_map
            global_type_map.update(type_map)
            debug(f"     gold_type_map: {type_map}")

        merged.append(record)
        n_ok += 1

    print(f"[INFO]  Entries with sexpr: {n_ok}, without/failed: {n_empty}")
    if has_types:
        n_types = sum(1 for e in merged if e.get("gold_type_map"))
        print(f"[INFO]  Entries with at least one type entity: {n_types}")

    out = paths["merged"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO]  Wrote {len(merged)} entries -> {out}")

    return global_type_map


def main():
    global _debug

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--kb")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run_config", type=str, help="Path to configs/run/<name>.yaml")

    apply_run_config_defaults(parser, section="labels")

    args = parser.parse_args()
    require(args, "dataset", "kb")

    _debug = args.debug
    
    if not ENDPOINT_URL:
        raise ValueError((
            "$(ENDPOINT_URL) is not set. Run using the Makefile "
            "to automatically set it to the run config's value."
            ))

    debug(f"ENDPOINT_URL (default) = {ENDPOINT_URL}")
    debug(f"BATCH_SIZE = {BATCH_SIZE}")

    print(f"[INFO] Loading KB: {args.kb}")
    kb = load_kb_module(args.kb)

    debug(f"KB LANGUAGE = {getattr(kb, 'LANGUAGE', '(not set)')}")
    debug(f"KB ENDPOINT_URL = {resolve_endpoint(kb)}")
    debug(f"KB LABEL_QUERY =\n{getattr(kb, 'LABEL_QUERY', '(not set)')}")

    paths_by_split = discover_paths(args.dataset)
    if not paths_by_split:
        print("no input files found", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found splits: {', '.join(paths_by_split.keys())}")

    has_types = hasattr(kb, "TYPES_QUERY") and hasattr(kb, "parse_type_results")

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    types_cache: dict[str, bool] = {}
    types_cache_path: Path | None = None

    if has_types:
        types_cache_path = data_dir / args.dataset / "cache" / "types.json"
        types_cache = load_cache(types_cache_path)
        debug(f"   Loaded types cache with {len(types_cache)} entries from {types_cache_path}")

    # Process each discovered split, and within it each mode
    global_type_map: dict[str, str] = {}

    for split, paths_by_mode in paths_by_split.items():
        print(f"\n[INFO] === Split: {split} ===")

        split_type_map: dict[str, str] = {}

        for mode, paths in paths_by_mode.items():
            print(f"\n[INFO] --- Mode: {mode} ---")

            cache = load_cache(paths["label_cache"])
            debug(f"   Loaded label cache with {len(cache)} entries from {paths['label_cache']}")

            dataset: list = []
            for inp in paths["inputs"]:
                print(f"[INFO]     Reading file: {inp}")
                dataset.extend(json.loads(inp.read_text(encoding="utf-8")))
            print(f"[INFO]     Total: {len(dataset)}")

            mode_type_map = action_merge_all(
                args.dataset, split, dataset, cache, kb, paths,
                types_cache=types_cache if has_types else None,
            )
            split_type_map.update(mode_type_map)
            global_type_map.update(mode_type_map)

            if has_types and types_cache_path is not None:
                save_cache(types_cache, types_cache_path)

        # Only write type map for train split
        if has_types and split == "train":
            type_map_path = (data_dir / args.dataset / "generation" / "label_maps" / f"{args.dataset}_{split}_type_label_map.json")
            type_map_path.parent.mkdir(parents=True, exist_ok=True)
            type_map_path.write_text(
                json.dumps(split_type_map, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\n[INFO] Wrote type label map for split '{split}' ({len(split_type_map)} entries) -> {type_map_path}")
    
    # Print failure report if in debug mode
    if _debug:
        print_failure_report()


if __name__ == "__main__":
    main()