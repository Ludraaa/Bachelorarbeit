import json
import re
import sys
import argparse
import os
from pathlib import Path
import requests
from utils.retry import call_with_retry
from src.utils.kb import load_kb_module
from src.utils.run_config import apply_run_config_defaults, require


ENDPOINT_URL = os.getenv("ENDPOINT_URL", "https://query.wikidata.org/sparql")
BATCH_SIZE = 50

SPLITS = ("dev", "test", "train")

_SPARQL_HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "insert_labels/1.0",
}

URI_PATTERN = re.compile(r"<(https?://[^>]+)>")

# ---------------------------------------------------------------------------
# debug stuff

_debug = False

def debug(*args, **kwargs):
    if _debug:
        print("[DEBUG]", *args, **kwargs)


# ---------------------------------------------------------------------------
# failure tracking

_label_failures: dict[str, dict] = {}  # URI -> failure info
_relation_failures: dict[str, dict] = {}


def log_label_failure(uri: str, reason: str, details: str = ""):
    """Track label resolution failures for final reporting."""
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
    """Track relation label resolution failures for final reporting."""
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
    """Print detailed failure report at the end."""
    if not _debug:
        return
    
    print("\n" + "="*80)
    print("LABEL RESOLUTION FAILURE REPORT")
    print("="*80)
    
    # Entity failures
    print(f"\n--- Entity Label Failures ({len(_label_failures)} unique URIs) ---")
    if not _label_failures:
        print("  None - all entities resolved successfully!")
    else:
        # Group by reason
        by_reason: dict[str, list] = {}
        for uri, info in _label_failures.items():
            by_reason.setdefault(info["reason"], []).append((uri, info))
        
        for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
            print(f"\n  Reason: {reason} ({len(items)} URIs)")
            # Show top 10 examples
            for uri, info in sorted(items, key=lambda x: -x[1]["count"])[:10]:
                print(f"    - {uri} (occurred {info['count']} times)")
                if info["details"]:
                    print(f"      Details: {info['details']}")
                if info["examples"]:
                    example_ids = info["examples"][:5]
                    print(f"      Example entries: {', '.join(example_ids)}")
    
    # Relation failures
    print(f"\n--- Relation Label Failures ({len(_relation_failures)} unique URIs) ---")
    if not _relation_failures:
        print("  None - all relations resolved successfully!")
    else:
        by_reason = {}
        for uri, info in _relation_failures.items():
            by_reason.setdefault(info["reason"], []).append((uri, info))
        
        for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
            print(f"\n  Reason: {reason} ({len(items)} URIs)")
            for uri, info in sorted(items, key=lambda x: -x[1]["count"])[:10]:
                print(f"    - {uri} (occurred {info['count']} times)")
                if info["details"]:
                    print(f"      Details: {info['details']}")
                if info["examples"]:
                    example_ids = info["examples"][:5]
                    print(f"      Example entries: {', '.join(example_ids)}")
    
    print("\n" + "="*80)


# ---------------------------------------------------------------------------
# endpoint resolution


def resolve_endpoint(kb) -> str:
    return getattr(kb, "LABEL_ENDPOINT_URL", None) or ENDPOINT_URL


# ---------------------------------------------------------------------------
# file stuff

def discover_paths(dataset: str) -> dict[str, dict[str, dict]]:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    base = data_dir / dataset
    input_dir = base / "sexpr"

    result: dict[str, dict[str, dict]] = {}

    for split in SPLITS:
        grouped: dict[str, list[Path]] = {}
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
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# labels


def fetch_labels_with_tracking(uris: list[str], cache: dict, kb) -> dict:
    """Enhanced fetch_labels with failure tracking."""
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
        
        norm = kb.normalize(uri)
        
        if norm is None:
            cache[uri] = None
            n_null += 1
            # Track normalize failures
            if "/entity/Q" in uri or "/entity/P" in uri:
                log_label_failure(uri, "normalize_failed", f"kb.normalize('{uri}') returned None")
            elif "/prop/" in uri:
                log_relation_failure(uri, "normalize_failed", f"kb.normalize('{uri}') returned None")
            debug(f"  normalize -> None (skipping SPARQL): {uri}")
            continue
        
        uri_map.setdefault(norm, []).append(uri)
        n_queued += 1
    
    debug(f"fetch_labels: {n_cached} already cached with labels, {n_null} no-label, "
          f"{n_queued} queued for SPARQL")
    
    missing = list(uri_map)
    if not missing:
        debug("  nothing to fetch")
        return cache
    
    endpoint = resolve_endpoint(kb)
    
    total_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  fetching {len(missing)} URIs in {total_batches} batch(es)")
    debug(f"  endpoint: {endpoint}")
    
    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i : i + BATCH_SIZE]
        batch_id = i // BATCH_SIZE + 1
        
        values = " ".join(f"<{u}>" for u in batch)
        query = kb.LABEL_QUERY.format(values=values, language=kb.LANGUAGE)
        
        debug(f"  batch {batch_id}/{total_batches}: {len(batch)} URIs")
        
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
                        log_label_failure(orig, "sparql_no_label", 
                                        f"SPARQL returned no label for normalized URI {norm}")
                    elif "/prop/" in orig:
                        log_relation_failure(orig, "sparql_no_label", 
                                           f"SPARQL returned no label for normalized URI {norm}")
        
        except Exception as e:
            print(f"  batch {batch_id}/{total_batches} failed: {e}")
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
# types  (only called when kb exposes TYPES_QUERY + parse_type_results)


def fetch_types(uris: list[str], types_cache: dict[str, bool], kb) -> dict[str, bool]:
    """
    Populate *types_cache* with True/False for each URI in *uris*.

    Requires kb.TYPES_QUERY (format string with {values}) and
    kb.parse_type_results(bindings) -> set[str].
    """
    missing = [u for u in uris if u not in types_cache]
    if not missing:
        return types_cache

    endpoint = resolve_endpoint(kb)

    total_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  fetching type membership for {len(missing)} URIs "
          f"in {total_batches} batch(es)")
    debug(f"  endpoint: {endpoint}")

    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i : i + BATCH_SIZE]
        batch_id = i // BATCH_SIZE + 1

        values = " ".join(f"<{u}>" for u in batch)
        query = kb.TYPES_QUERY.format(values=values)

        debug(f"  types batch {batch_id}/{total_batches}: {len(batch)} URIs")

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
            bindings = resp.json()["results"]["bindings"]
            type_uris = kb.parse_type_results(bindings)
            debug(f"  parse_type_results -> {len(type_uris)} type URI(s)")

            for u in batch:
                types_cache[u] = (u in type_uris)

        except Exception as e:
            print(f"  types batch {batch_id}/{total_batches} failed: {e}")
            if _debug:
                import traceback
                traceback.print_exc()
            # Mark as False so we don't retry forever on a broken endpoint
            for u in batch:
                if u not in types_cache:
                    types_cache[u] = False

    return types_cache


def apply_labels_with_tracking(sexpr: str, cache: dict, kb, entry_id: str = "") -> str:
    """Enhanced version of apply_labels that tracks failures."""
    formatter = getattr(kb, "format_label", None)

    def repl(match: re.Match) -> str:
        uri = match.group(1)
        label = cache.get(uri)

        # Track failures
        if label is None:
            # Check why it failed
            if uri not in cache:
                reason = "not_in_cache"
                details = "URI was never fetched from SPARQL"
            else:
                reason = "null_label"
                details = "SPARQL returned no label (normalize returned None or no label found)"
            
            if "/entity/Q" in uri or "/entity/P" in uri:
                log_label_failure(uri, reason, details)
                if entry_id and entry_id not in _label_failures.get(uri, {}).get("examples", []):
                    _label_failures.setdefault(uri, {}).setdefault("examples", []).append(entry_id)
            elif "/prop/" in uri:
                log_relation_failure(uri, reason, details)
                if entry_id and entry_id not in _relation_failures.get(uri, {}).get("examples", []):
                    _relation_failures.setdefault(uri, {}).setdefault("examples", []).append(entry_id)
            
            debug(f"  apply_labels: {uri} -> NO LABEL ({reason})")
            return match.group(0)

        # Success case
        debug(f"  apply_labels: {uri} -> label={label!r}")

        if formatter:
            out = formatter(uri, label or "")
            if out:
                return out

        if label:
            return label.replace(" ", "_")

        return match.group(0)

    return URI_PATTERN.sub(repl, sexpr)


# ---------------------------------------------------------------------------
# processing


def action_merge_all(
    dataset_name: str,
    split: str,
    dataset: list,
    cache: dict,
    kb,
    paths: dict,
    types_cache: dict[str, bool] | None = None,
) -> dict[str, str]:

    has_types = (
        types_cache is not None
        and hasattr(kb, "TYPES_QUERY")
        and hasattr(kb, "parse_type_results")
    )

    format_rel = getattr(kb, "format_relation_label", None)

    print("  collecting URIs...")
    all_uris: set[str] = set()

    for entry in dataset:
        s = entry.get("Sexpr", "")
        if s and s != "Parsing failed":
            all_uris.update(URI_PATTERN.findall(s))

    missing_labels = [u for u in all_uris if u not in cache]
    print(f"  {len(all_uris)} URIs ({len(missing_labels)} new)")
    debug(f"  sample URIs from sexprs: {sorted(all_uris)[:5]}")

    if missing_labels:
        fetch_labels_with_tracking(list(all_uris), cache, kb)
        save_cache(cache, paths["label_cache"])

        n_resolved = sum(1 for u in all_uris if cache.get(u))
        n_none     = sum(1 for u in all_uris if u in cache and cache[u] is None)
        debug(f"  after fetch: {n_resolved} resolved, {n_none} null, "
              f"{len(all_uris) - n_resolved - n_none} still missing")

    # Fetch type membership
    if has_types:
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
            debug(f"  type membership already cached for all {len(all_entity_uris)} entity URIs")

    print("  building output...")
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

        if not sexpr or sexpr == "Parsing failed":
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

        debug(f"  entry {entry.get('id', '?')}: "
              f"{len(entities)} entities, {len(relations)} relations")

        ent_map = {u: cache[u] for u in entities if cache.get(u)}
        rel_map = {
            u: (format_rel(u, cache[u]) if format_rel else cache[u])
            for u in relations if u in cache
        }

        if _debug:
            unresolved_ents = [u for u in entities if not cache.get(u)]
            unresolved_rels = [u for u in relations if not cache.get(u)]
            if unresolved_ents or unresolved_rels:
                debug(f"    unresolved entities: {unresolved_ents}")
                debug(f"    unresolved relations: {unresolved_rels}")
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
            "sexpr_with_labels": apply_labels_with_tracking(sexpr, cache, kb, entry.get("id", "")),
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
            debug(f"    gold_type_map: {type_map}")

        merged.append(record)
        n_ok += 1

    print(f"  entries with sexpr: {n_ok}, without/failed: {n_empty}")
    if has_types:
        n_types = sum(1 for e in merged if e.get("gold_type_map"))
        print(f"  entries with at least one type entity: {n_types}")

    out = paths["merged"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {len(merged)} entries -> {out}")

    return global_type_map


def main():
    global _debug

    parser = argparse.ArgumentParser()
    # NOTE: was required=True. Changed to optional + require() below so a
    # run_config's top-level `dataset` can fill it in.
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--kb", default="wikidata")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run_config", type=str, default=None,
                        help="Path to configs/run/<name>.yaml; values become defaults, "
                             "explicit flags still override.")

    apply_run_config_defaults(parser, section="labels")

    args = parser.parse_args()
    require(args, "dataset")

    _debug = args.debug

    debug(f"ENDPOINT_URL (default) = {ENDPOINT_URL}")
    debug(f"BATCH_SIZE             = {BATCH_SIZE}")

    print(f"loading KB: {args.kb}")
    kb = load_kb_module(args.kb)

    debug(f"KB LANGUAGE    = {getattr(kb, 'LANGUAGE', '(not set)')}")
    debug(f"KB ENDPOINT_URL = {resolve_endpoint(kb)}")
    debug(f"KB LABEL_QUERY =\n{getattr(kb, 'LABEL_QUERY', '(not set)')}")

    paths_by_split = discover_paths(args.dataset)
    if not paths_by_split:
        print("no input files found", file=sys.stderr)
        sys.exit(1)

    print(f"Found splits: {', '.join(paths_by_split.keys())}")

    has_types = hasattr(kb, "TYPES_QUERY") and hasattr(kb, "parse_type_results")

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    types_cache: dict[str, bool] = {}
    types_cache_path: Path | None = None

    if has_types:
        types_cache_path = data_dir / args.dataset / "cache" / "types.json"
        types_cache = load_cache(types_cache_path)
        debug(f"loaded types cache with {len(types_cache)} entries "
              f"from {types_cache_path}")

    # Process each discovered split, and within it each mode
    global_type_map: dict[str, str] = {}

    for split, paths_by_mode in paths_by_split.items():
        print(f"\n=== split: {split} ===")

        split_type_map: dict[str, str] = {}

        for mode, paths in paths_by_mode.items():
            print(f"\n--- mode: {mode} ---")

            cache = load_cache(paths["label_cache"])
            debug(f"loaded label cache with {len(cache)} entries "
                  f"from {paths['label_cache']}")

            dataset: list = []
            for inp in paths["inputs"]:
                print(f"  reading {inp}")
                dataset.extend(json.loads(inp.read_text(encoding="utf-8")))
            print(f"  total: {len(dataset)}")

            mode_type_map = action_merge_all(
                args.dataset, split, dataset, cache, kb, paths,
                types_cache=types_cache if has_types else None,
            )
            split_type_map.update(mode_type_map)
            global_type_map.update(mode_type_map)

            # Persist the types cache after each mode so a crash doesn't lose work.
            if has_types and types_cache_path is not None:
                save_cache(types_cache, types_cache_path)

        if has_types and split == "train":
            type_map_path = (
                data_dir / args.dataset / "generation" / "label_maps"
                / f"{args.dataset}_{split}_type_label_map.json"
            )
            type_map_path.parent.mkdir(parents=True, exist_ok=True)
            type_map_path.write_text(
                json.dumps(split_type_map, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\nwrote type label map for split '{split}' "
                  f"({len(split_type_map)} entries) -> {type_map_path}")

    print("\ndone")
    
    # Print failure report if in debug mode
    if _debug:
        print_failure_report()


if __name__ == "__main__":
    main()