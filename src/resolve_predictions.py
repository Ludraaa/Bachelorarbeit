import os
import re
import sys
import json
import time
import argparse
import importlib.util
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product as iter_product
import heapq
from pathlib import Path
from typing import Any

import psutil
import requests
from tqdm import tqdm

from linkers import (
    load_extractor,
    load_substitute,
    load_entity_linker,
    load_predicate_linker,
)
from linkers.base import LinkingInput, LinkingOutput
from sexpr.jena_interface import algebra_to_sparql
from src.utils.retry import call_with_retry
from src.legacy.lisp_to_sparql_chatkbqa import sexpr_to_sparql as chatkbqa_webqsp_sexpr_to_sparql
from src.legacy.lisp_to_sparql_chatkbqa_cwq import sexpr_to_sparql as chatkbqa_cwq_sexpr_to_sparql
from src.utils.sparql_exec import _SPARQL_HEADERS
from src.utils.kb import load_kb_module

ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "https://query.wikidata.org/sparql")


_PROC = psutil.Process(os.getpid())

DO_LOG = False

def _ram() -> str:
    rss = _PROC.memory_info().rss / 1024**3
    return f"{rss:.2f} GB RSS"

def _log(msg: str) -> None:
    if DO_LOG:
        print(f"[LOG {_ram()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Per-pass parameter helpers

def _parse_ints(s: str, fallback: int) -> list[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    return vals if vals else [fallback]

def _parse_floats(s: str, fallback: float) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    return vals if vals else [fallback]

def _get_pass_val(values: list, pass_idx: int):
    return values[pass_idx] if pass_idx < len(values) else values[-1]


# ---------------------------------------------------------------------------
# Per-item timeout helper

def _deadline_exceeded(deadline: float | None) -> bool:
    """
    `deadline` is an absolute time.perf_counter() timestamp, or None for
    "no limit". Checked at cheap, regular intervals inside the beam /
    entity-permutation / predicate-permutation / SPARQL-candidate loops so
    a slow item can be abandoned promptly instead of running to completion.
    """
    return deadline is not None and time.perf_counter() > deadline


# ---------------------------------------------------------------------------
# Args

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",   type=str, required=True)
    parser.add_argument("--split",     type=str, default="test")
    parser.add_argument(
        "--mode", type=str, default="simple",
        choices=["chatkbqa_webqsp", "chatkbqa_cwq", "jena", "sparql"],
    )
    parser.add_argument("--model_id",  type=str, required=True)
    parser.add_argument("--data_dir",  type=str, default="data")

    parser.add_argument(
        "--entity_linkers",
        type=str,
        default="wikidata_sim",
        help=(
            "Comma-separated ordered list of entity linker IDs. Each linker "
            "only sees labels still unresolved by the ones before it, "
            "mirroring ChatKBQA's type_map -> gold -> SimCSE -> FACC1 cascade. "
            "Example: --entity_linkers type_map,gold_exact,gold_simcse,facc1"
        ),
    )

    parser.add_argument(
        "--predicate_linkers",
        type=str,
        default="label_search",
        help=(
            "Comma-separated ordered list of predicate linker IDs. "
            "Each item is tried across all beams before the next is attempted. "
            "Example: --predicate_linkers label_norm,neighborhood_simcse"
        ),
    )

    parser.add_argument("--kb", type=str, default="wikidata")

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)


    parser.add_argument("--k1_per_pass", type=str, default="25",
                        help="Comma-separated k1 per predicate-linker pass (single value broadcast to all passes).")
    parser.add_argument("--t1_per_pass", type=str, default="0.0",
                        help="Comma-separated t1 per predicate-linker pass (single value broadcast to all passes).")
    parser.add_argument("--k2_per_pass", type=str, default="5",
                        help="Comma-separated k2 per predicate-linker pass (single value broadcast to all passes).")
    parser.add_argument("--t2_per_pass", type=str, default="0.0",
                        help="Comma-separated t2 per predicate-linker pass (single value broadcast to all passes).")

    parser.add_argument(
        "--beam_limits",
        type=str,
        help=(
            "Comma-separated per-pass beam caps, one per predicate linker "
            "(use 0 for no limit, last value is reused). "
            "Example: --beam_limits 5,3"
        ),
    )

    parser.add_argument(
        "--linker_params",
        type=str,
        default="{}",
        help=(
            'JSON dict overriding constructor kwargs per linker id, e.g. '
            '\'{"ChatKBQA.gold_simcse": {"gold_threshold": 0.5}}\'. '
            'Applies to both entity and predicate linkers by id.'
        ),
    )

    parser.add_argument(
        "--item_time_limit_sec",
        type=float,
        default=None,
        help=(
            "Optional wall-clock time budget per item, in seconds, shared "
            "across all beams and all predicate-linker passes for that item. "
            "If exceeded, the item is abandoned immediately (no further "
            "beams/passes are tried), recorded as unresolved with "
            "timed_out=true, and processing moves on to the next item. "
            "Default: no limit."
        ),
    )

    parser.add_argument(
        "--note", type=str, default="",
        help="Optional free-text note stored in the output metadata.",
    )

    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# File stuff

def load_predictions(data_dir, dataset, model_id, split, mode):
    path = os.path.join(
        data_dir, dataset, "predictions", model_id, "raw",
        f"{dataset}_{split}.{mode}.json",
    )
    print(f"Loading predictions from: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
        meta = data.get("meta", {})
        items = data.get("items", {})
    print(f"Loaded {len(items)} items")
    return [items, meta]


def resolve_output_path(args, linker_id: str) -> str:
    out_dir = args.output_dir or os.path.join(
        args.data_dir, args.dataset, "predictions",
        args.model_id, "resolved", linker_id,
    )
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{args.dataset}_{args.split}.{args.mode}.jsonl")


# ---------------------------------------------------------------------------
# JSONL helpers

def _load_existing_jsonl(path: str) -> tuple[list[dict], int]:
    """
    Read an existing JSONL file.  Silently skips corrupt / partial lines
    (e.g. a half-written line from a previous crash).
    Returns (items, valid_line_count).
    """
    items: list[dict] = []
    if not os.path.exists(path):
        return items, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip corrupt tail line from a previous crash
    return items, len(items)


def _append_jsonl(path: str, obj: dict) -> None:
    """Append a single JSON object as a line to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _finalize_to_json(jsonl_path: str, meta: dict) -> str:
    """
    Read completed JSONL, combine with meta, write pretty JSON.
    Returns the JSON path.
    """
    json_path = jsonl_path.replace(".jsonl", ".json")
    items, _ = _load_existing_jsonl(jsonl_path)
    output = {"meta": meta, "items": items}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return json_path


# ---------------------------------------------------------------------------
# Permutation

def _kbest_cartesian(
    candidates: dict[str, list[tuple[str, float]]],
    k: int,
    t: float,
) -> list[tuple[dict[str, str], float]]:
    labels     = list(candidates.keys())
    cand_lists = [candidates[lbl] for lbl in labels]
    if not labels or not all(cand_lists):
        return []

    n = len(labels)
    sorted_lists = [sorted(lst, key=lambda x: -x[1]) for lst in cand_lists]

    def mean_score(indices: tuple) -> float:
        return sum(sorted_lists[i][idx][1] for i, idx in enumerate(indices)) / n

    init = (0,) * n
    heap: list[tuple[float, tuple]] = [(-mean_score(init), init)]
    visited: set[tuple] = {init}
    results: list[tuple[dict[str, str], float]] = []

    while heap and len(results) < k:
        neg_s, indices = heapq.heappop(heap)
        label_map = {labels[i]: sorted_lists[i][idx][0] for i, idx in enumerate(indices)}
        results.append((label_map, -neg_s))

        for i in range(n):
            if indices[i] + 1 >= len(sorted_lists[i]):
                continue
            neighbour = indices[:i] + (indices[i] + 1,) + indices[i + 1:]
            if neighbour not in visited:
                visited.add(neighbour)
                heapq.heappush(heap, (-mean_score(neighbour), neighbour))

    filtered = [(m, s) for m, s in results if s >= t]
    return filtered if filtered else results


def permute_by_entity(
    entity_candidates: dict[str, list[tuple[str, float]]],
    k1: int,
    t1: float,
) -> list[tuple[dict[str, str], float]]:
    return _kbest_cartesian(entity_candidates, k1, t1)


def permute_by_relation(
    predicate_candidates: dict[str, list[tuple[str, float]]],
    k2: int,
    t2: float,
) -> list[tuple[dict[str, str], float]]:
    return _kbest_cartesian(predicate_candidates, k2, t2)


# ---------------------------------------------------------------------------
# Entity linker chain

def run_entity_linker_chain(
    entity_linkers: list,
    entity_linker_ids: list[str],
    labels: list[str],
    question: str,
    beam: str,
    item: dict,
    type_map: dict,
    debug: bool,
) -> tuple[LinkingOutput, list[dict] | None]:
    unresolved = list(labels)
    label_map: dict[str, str] = {}
    candidates: dict[str, list] = {}
    chain_debug = [] if debug else None

    for linker, linker_id in zip(entity_linkers, entity_linker_ids):
        if not unresolved:
            break

        out = linker.link(LinkingInput(
            labels=unresolved,
            question=question,
            prediction=beam,
            item=item,
            type_map=type_map,
        ))

        resolved_now = []
        for label in unresolved:
            cands = out.candidates.get(label) or []
            if cands:
                candidates[label] = cands
                if label in out.label_map:
                    label_map[label] = out.label_map[label]
                resolved_now.append(label)

        if debug:
            chain_debug.append({
                "linker_id": linker_id,
                "attempted": list(unresolved),
                "resolved": resolved_now,
            })

        unresolved = [l for l in unresolved if l not in resolved_now]

    for label in unresolved:
        candidates.setdefault(label, [])

    e_out = LinkingOutput(
        label_map=label_map,
        candidates=candidates,
        failed=unresolved,
        debug={},
    )
    return e_out, chain_debug


# ---------------------------------------------------------------------------
# Single-pass resolution logic

@dataclass
class PassResult:
    found: bool  = False
    executed_query: str | None = None
    used_beam_rank: int | None = None
    entity_map_used: dict = field(default_factory=dict)
    predicate_map_used: dict = field(default_factory=dict)
    entity_perm_idx: int | None = None
    predicate_perm_idx: int | None = None
    pass_index: int = -1
    pass_linker_id: str = ""
    beam_debug: list = field(default_factory=list)
    runtime_sec: float = 0.0
    timed_out: bool = False


def run_single_pass(
    *,
    beams: list[str],
    question: str,
    item: dict,
    entity_linkers: list,
    entity_linker_ids: list[str],
    predicate_linker,
    k1: int,
    t1: float,
    k2: int,
    t2: float,
    common_prefixes: dict,
    extract,
    substitute,
    type_map: dict,
    mode: str,
    beam_limit: int,
    pass_index: int,
    pass_linker_id: str,
    debug: bool,
    deadline: float | None = None,
) -> PassResult:
    pass_start = time.perf_counter()
    result = PassResult(pass_index=pass_index, pass_linker_id=pass_linker_id)

    beams_to_try = beams if not beam_limit else beams[:beam_limit]

    for beam_rank, beam in enumerate(beams_to_try):
        if result.found or result.timed_out:
            break

        if _deadline_exceeded(deadline):
            _log(f"  beam {beam_rank}: item time limit exceeded, aborting pass '{pass_linker_id}'")
            result.timed_out = True
            break

        _log(f"  beam {beam_rank}: extract labels")
        entity_labels, predicate_labels = extract(beam)
        _log(f"  beam {beam_rank}: entity_labels={entity_labels} predicate_labels={predicate_labels}")

        _log(f"  beam {beam_rank}: entity linker chain (n_entity_labels={len(entity_labels)})")
        e_out, entity_chain_debug = run_entity_linker_chain(
            entity_linkers=entity_linkers,
            entity_linker_ids=entity_linker_ids,
            labels=entity_labels,
            question=question,
            beam=beam,
            item=item,
            type_map=type_map,
            debug=debug,
        )
        _log(f"  beam {beam_rank}: entity linking done, n_candidates={sum(len(v) for v in e_out.candidates.values())}")

        _log(f"  beam {beam_rank}: permute_by_entity (k1={k1}, t1={t1})")
        entity_permutations = permute_by_entity(e_out.candidates, k1, t1)
        _log(f"  beam {beam_rank}: {len(entity_permutations)} entity permutations")

        if not entity_permutations and e_out.label_map:
            entity_permutations = [(e_out.label_map, 0.0)]
        elif not entity_permutations:
            entity_permutations = [({}, 1.0)]

        beam_debug_entry: dict[str, Any] = {}
        if debug:
            beam_debug_entry = {
                "rank": beam_rank,
                "raw_beam": beam,
                "pass_index": pass_index,
                "pass_linker_id": pass_linker_id,
                "entity_labels": entity_labels,
                "predicate_labels": predicate_labels,
                "entity_chain": entity_chain_debug,
                "entity_candidates": e_out.candidates,
                "entity_failed": e_out.failed,
                "entity_permutations": [
                    {"entity_map": em, "score": s}
                    for em, s in entity_permutations
                ],
                "relation_permutations_tried": [],
                "predicate_debug": [],
            }

        for ep_idx, (entity_map, ep_score) in enumerate(entity_permutations):
            if result.found or result.timed_out:
                break

            if _deadline_exceeded(deadline):
                _log(f"  beam {beam_rank}, ep {ep_idx}: item time limit exceeded, aborting pass '{pass_linker_id}'")
                result.timed_out = True
                break

            _log(f"  beam {beam_rank}, ep {ep_idx}/{len(entity_permutations)}: predicate_linker.link (n_pred_labels={len(predicate_labels)})")
            p_out = predicate_linker.link(
                LinkingInput(
                    labels=predicate_labels,
                    question=question,
                    prediction=beam,
                    item=item,
                ),
                entity_map=entity_map,
            )
            _log(f"  beam {beam_rank}, ep {ep_idx}: predicate linking done, n_candidates={sum(len(v) for v in p_out.candidates.values())}")

            if debug:
                beam_debug_entry["predicate_debug"].append({
                    "entity_perm_idx": ep_idx,
                    "entity_map": entity_map,
                    "per_label": p_out.debug,
                })

            _log(f"  beam {beam_rank}, ep {ep_idx}: permute_by_relation (k2={k2}, t2={t2})")
            predicate_permutations = permute_by_relation(p_out.candidates, k2, t2)
            _log(f"  beam {beam_rank}, ep {ep_idx}: {len(predicate_permutations)} predicate permutations")

            if not predicate_permutations and p_out.label_map:
                predicate_permutations = [(p_out.label_map, 0.0)]

            for pp_idx, (predicate_map, pp_score) in enumerate(predicate_permutations):
                if result.found or result.timed_out:
                    break

                if _deadline_exceeded(deadline):
                    _log(f"  beam {beam_rank}, ep {ep_idx}, pp {pp_idx}: item time limit exceeded, aborting pass '{pass_linker_id}'")
                    result.timed_out = True
                    break

                _log(f"  beam {beam_rank}, ep {ep_idx}, pp {pp_idx}: substitute + to_sparql")
                resolved = substitute(beam, entity_map, predicate_map, True)

                sparql_candidates = to_sparql(resolved, mode)
                sparql_candidates = [inject_prefixes(s, common_prefixes) for s in sparql_candidates]
                conversion_ok = len(sparql_candidates) > 0

                _log(f"beam:{beam}")
                _log(f"entities: {entity_map}")
                _log(f"predicates: {predicate_map}")
                _log(f"substituted:\n{resolved}")

                bindings         = None
                exec_ok          = False
                has_results      = False
                sparql_candidate = sparql_candidates[0] if sparql_candidates else None
                candidates_tried = []
                cand_loop_timed_out = False

                for cand_idx, cand_sparql in enumerate(sparql_candidates):
                    if _deadline_exceeded(deadline):
                        _log(f"  beam {beam_rank}, ep {ep_idx}, pp {pp_idx}: item time limit exceeded before SPARQL candidate {cand_idx}")
                        cand_loop_timed_out = True
                        break

                    _log(f"  beam {beam_rank}, ep {ep_idx}, pp {pp_idx}: execute_sparql (candidate {cand_idx})")
                    _log(f"  sparql candidate {cand_idx}: \n{cand_sparql}")
                    cand_bindings    = execute_sparql(cand_sparql)
                    cand_exec_ok     = cand_bindings is not None
                    cand_has_results = _has_results(cand_bindings)
                    _log(f"  beam {beam_rank}, ep {ep_idx}, pp {pp_idx}: candidate {cand_idx} exec_ok={cand_exec_ok} has_results={cand_has_results}")

                    if debug:
                        candidates_tried.append({
                            "candidate_index": cand_idx,
                            "sparql": cand_sparql,
                            "exec_ok": cand_exec_ok,
                            "has_results": cand_has_results,
                        })

                    bindings         = cand_bindings
                    exec_ok          = cand_exec_ok
                    has_results      = cand_has_results
                    sparql_candidate = cand_sparql

                    if cand_has_results:
                        break

                if debug:
                    beam_debug_entry["relation_permutations_tried"].append({
                        "entity_perm_idx": ep_idx,
                        "entity_map": entity_map,
                        "entity_perm_score": ep_score,
                        "predicate_perm_idx": pp_idx,
                        "predicate_map": predicate_map,
                        "predicate_perm_score": pp_score,
                        "resolved_query": resolved,
                        "sparql": sparql_candidate,
                        "sparql_candidates_tried": candidates_tried,
                        "conversion_ok": conversion_ok,
                        "exec_ok": exec_ok,
                        "has_results": has_results,
                        "timed_out": cand_loop_timed_out,
                    })

                if cand_loop_timed_out:
                    result.timed_out = True
                    break

                if has_results:
                    result.found = True
                    result.executed_query = sparql_candidate
                    result.used_beam_rank = beam_rank
                    result.entity_map_used = entity_map
                    result.predicate_map_used = predicate_map
                    result.entity_perm_idx = ep_idx
                    result.predicate_perm_idx = pp_idx

        if debug and beam_debug_entry:
            result.beam_debug.append(beam_debug_entry)

    result.runtime_sec = time.perf_counter() - pass_start
    return result


# ---------------------------------------------------------------------------
# SPARQL stuff

def _entity_label_fallback(sparql: str) -> str | None:
    """
    Mirrors ChatKBQA's own zero-result retry in aggressive_top_k_eval_new.py's
    execute_normed_s_expr_from_label_maps()
    """
    entities = sorted(set(re.findall(r'\bns:(m\.[A-Za-z0-9_]+)\b', sparql)))
    if not entities:
        return None

    addlines = []
    rewritten = sparql
    for i, ent in enumerate(entities):
        var = f"?ei{i}"
        addlines.append(f'ns:{ent} rdfs:label ?en{i} . ')
        addlines.append(f'{var} rdfs:label ?en{i} . ')
        addlines.append(f'FILTER (langMatches( lang(?en{i}), "EN" ) )')
        rewritten = rewritten.replace(f'ns:{ent}', var)

    anchor = "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))"
    lines = rewritten.split('\n')
    for idx, line in enumerate(lines):
        if line.strip() == anchor:
            lines = lines[:idx + 1] + addlines + lines[idx + 1:]
            return '\n'.join(lines)
    return None


def to_sparql(query: str, mode: str) -> list[str]:
    if mode == "sparql":
        return [query]

    if mode == "jena":
        try:
            sparql = algebra_to_sparql(query)
        except Exception:
            return []
        return [sparql] if sparql is not None else []

    if mode == "chatkbqa_webqsp":
        try:
            sparql = chatkbqa_webqsp_sexpr_to_sparql(query)
        except Exception:
            return []
        if sparql is None:
            return []
        candidates = [sparql]
        #fallback = _entity_label_fallback(sparql)
        #if fallback is not None:
        #    candidates.append(fallback)
        return candidates

    if mode == "chatkbqa_cwq":
        try:
            sparql = chatkbqa_cwq_sexpr_to_sparql(query)
        except Exception:
            return []
        if sparql is None:
            return []
        candidates = [sparql]
        #fallback = _entity_label_fallback(sparql)
        #if fallback is not None:
        #    candidates.append(fallback)
        return candidates

    # fallback
    return [query]


def inject_prefixes(sparql: str, common_prefixes: dict[str, str]) -> str:
    for prefix, uri in common_prefixes.items():
        declaration = f"PREFIX {prefix}:"
        if re.search(rf'\b{re.escape(prefix)}:[A-Za-z0-9_]', sparql) and declaration not in sparql:
            sparql = f"{declaration} <{uri}>\n{sparql}"
    return sparql


def execute_sparql(sparql: str) -> list | None:
    # only care about existence of results
    if not re.search(r'\bLIMIT\b', sparql, re.IGNORECASE):
        sparql += "\nLIMIT 10"

    def _do_request():
        resp = requests.post(
            ENDPOINT_URL,
            data={"query": sparql},
            headers=_SPARQL_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        return resp

    try:
        resp = call_with_retry(
            _do_request,
            retries=0,
            base_delay=1.0,
            backoff=2.0,
            exceptions=(requests.RequestException,),
            on_fail=None,
        )

        if resp is None:
            return None

        data = resp.json()

        if "bindings" in data.get("results", {}):
            return data["results"]["bindings"]

        if "boolean" in data:
            return [{"boolean": {"value": str(data["boolean"]).lower()}}]
        return None

    except (ValueError, KeyError):
        return None


def _has_results(bindings: list | None) -> bool:
    return bool(bindings)


# ---------------------------------------------------------------------------
# Multi-pass resolution for one item

def resolve_item(
    *,
    beams: list[str],
    question: str,
    item: dict,
    entity_linkers: list,
    entity_linker_ids: list[str],
    predicate_linkers: list,
    predicate_linker_ids: list[str],
    beam_limits: list[int],
    k1_list: list[int],
    t1_list: list[float],
    k2_list: list[int],
    t2_list: list[float],
    common_prefixes: dict,
    extract,
    substitute,
    mode: str,
    type_map: dict,
    debug: bool,
    time_limit_sec: float | None = None,
) -> tuple[PassResult, list[PassResult]]:
    all_pass_results: list[PassResult] = []

    item_deadline = (
        time.perf_counter() + time_limit_sec if time_limit_sec is not None else None
    )

    for pass_idx, (pred_linker, linker_id) in enumerate(
        zip(predicate_linkers, predicate_linker_ids)
    ):
        if _deadline_exceeded(item_deadline):
            _log(f"  pass {pass_idx} ({linker_id}): item time limit already exceeded, skipping remaining passes")
            all_pass_results.append(PassResult(
                pass_index=pass_idx,
                pass_linker_id=linker_id,
                timed_out=True,
                runtime_sec=0.0,
            ))
            break

        beam_limit = _get_pass_val(beam_limits, pass_idx)
        k1 = _get_pass_val(k1_list, pass_idx)
        t1 = _get_pass_val(t1_list, pass_idx)
        k2 = _get_pass_val(k2_list, pass_idx)
        t2 = _get_pass_val(t2_list, pass_idx)

        _log(f"  pass {pass_idx} ({linker_id}): starting  beam_limit={beam_limit}  k1={k1} t1={t1} k2={k2} t2={t2}")
        pass_result = run_single_pass(
            beams=beams,
            question=question,
            item=item,
            entity_linkers=entity_linkers,
            entity_linker_ids=entity_linker_ids,
            predicate_linker=pred_linker,
            k1=k1,
            t1=t1,
            k2=k2,
            t2=t2,
            common_prefixes=common_prefixes,
            extract=extract,
            substitute=substitute,
            mode=mode,
            type_map=type_map,
            beam_limit=beam_limit,
            pass_index=pass_idx,
            pass_linker_id=linker_id,
            debug=debug,
            deadline=item_deadline,
        )
        _log(f"  pass {pass_idx} ({linker_id}): done, found={pass_result.found}, timed_out={pass_result.timed_out}, runtime={pass_result.runtime_sec:.2f}s")

        all_pass_results.append(pass_result)

        if pass_result.found:
            return pass_result, all_pass_results

        if pass_result.timed_out:
            _log(f"  item time limit exceeded during pass {pass_idx} ({linker_id}); abandoning item, skipping remaining passes")
            break

    return all_pass_results[-1], all_pass_results


# ---------------------------------------------------------------------------
# Runtime aggregation helpers

def _new_runtime_agg(predicate_linker_ids: list[str]) -> dict:
    agg = {"total_count": 0, "total_sec": 0.0, "by_resolution": {}}
    for lid in predicate_linker_ids:
        agg["by_resolution"][lid] = {"count": 0, "total_sec": 0.0}
    agg["by_resolution"]["_unresolved"] = {"count": 0, "total_sec": 0.0}
    agg["by_resolution"]["_timeout"] = {"count": 0, "total_sec": 0.0}
    return agg


def _record_runtime(
    agg: dict,
    item_runtime_sec: float,
    winning_pass_linker: str | None,
    timed_out: bool = False,
) -> None:
    agg["total_count"] += 1
    agg["total_sec"] += item_runtime_sec
    if timed_out:
        key = "_timeout"
    elif winning_pass_linker is not None:
        key = winning_pass_linker
    else:
        key = "_unresolved"
    bucket = agg["by_resolution"].setdefault(key, {"count": 0, "total_sec": 0.0})
    bucket["count"] += 1
    bucket["total_sec"] += item_runtime_sec


def _runtime_summary(agg: dict) -> dict:
    avg_per_item = (agg["total_sec"] / agg["total_count"]) if agg["total_count"] else None
    by_resolution = {}
    for key, bucket in agg["by_resolution"].items():
        avg = (bucket["total_sec"] / bucket["count"]) if bucket["count"] else None
        by_resolution[key] = {
            "count": bucket["count"],
            "total_sec": round(bucket["total_sec"], 4),
            "avg_sec": round(avg, 4) if avg is not None else None,
        }
    return {
        "avg_runtime_sec_per_item": round(avg_per_item, 4) if avg_per_item is not None else None,
        "total_runtime_sec": round(agg["total_sec"], 4),
        "runtime_by_resolution": by_resolution,
    }


# ---------------------------------------------------------------------------
# Main

def main():
    args = parse_args()
    if args.debug:
        global DO_LOG
        DO_LOG = True

    entity_linker_ids = [s.strip() for s in args.entity_linkers.split(",") if s.strip()]
    predicate_linker_ids = [s.strip() for s in args.predicate_linkers.split(",") if s.strip()]
    n_passes = len(predicate_linker_ids)

    k1_list = _parse_ints(args.k1_per_pass, fallback=25)
    t1_list = _parse_floats(args.t1_per_pass, fallback=0.0)
    k2_list = _parse_ints(args.k2_per_pass, fallback=5)
    t2_list = _parse_floats(args.t2_per_pass, fallback=0.0)

    linker_combo_id = f"{'+'.join(entity_linker_ids)}+{'+'.join(predicate_linker_ids)}"

    kb_module = load_kb_module(args.kb)
    common_prefixes = getattr(kb_module, "COMMON_PREFIXES", {})

    try:
        linker_params = json.loads(args.linker_params)
    except json.JSONDecodeError as e:
        raise ValueError(f"--linker_params is not valid JSON: {e}") from e

    unknown_ids = set(linker_params) - set(entity_linker_ids) - set(predicate_linker_ids)
    if unknown_ids:
        raise ValueError(
            f"--linker_params references linker id(s) not in this run: {sorted(unknown_ids)}"
        )

    extract = load_extractor(args.kb)
    substitute  = load_substitute(args.kb)
    entity_linkers = [
        load_entity_linker(lid, **linker_params.get(lid, {})) for lid in entity_linker_ids
    ]
    predicate_linkers = [
        load_predicate_linker(lid, **linker_params.get(lid, {})) for lid in predicate_linker_ids
    ]

    entity_linker_params = {
        lid: linker.get_params() for lid, linker in zip(entity_linker_ids, entity_linkers)
    }
    predicate_linker_params = {
        lid: linker.get_params() for lid, linker in zip(predicate_linker_ids, predicate_linkers)
    }

    data, meta = load_predictions(
        args.data_dir, args.dataset, args.model_id, args.split, args.mode,
    )        

    if meta.get("max_beams") is not None and args.beam_limits is None:
        print("Found max beams in prediction file")
        beam_limits = [meta.get("max_beams", 0)]
    elif args.beam_limits:
        beam_limits = _parse_ints(args.beam_limits, fallback=0)
        print("Beam limit explicitly passed.")
    else:
        raise ValueError(
            "No metadata found in prediction file, please explicitly pass --beam_limits"
        )

    # Load the inverted type label map if one exists for this KB/dataset/split
    _type_map_path = Path(args.data_dir) / args.dataset / "generation" / "label_maps" \
                    / f"{args.dataset}_train_type_label_map.json"
    if _type_map_path.exists():
        _raw_type_map = json.loads(_type_map_path.read_text(encoding="utf-8"))
        type_map = {label.lower(): mid.split("/")[-1] for mid, label in _raw_type_map.items()}
        print(f"Loaded type label map: {len(type_map)} entries from {_type_map_path}")
    else:
        type_map = {}
        print("No type label map found — type-first resolution disabled")

    if args.max_samples:
        data = data[: args.max_samples]
        print(f"Capped to {len(data)} examples")

    jsonl_path = resolve_output_path(args, linker_combo_id)
    json_path  = jsonl_path.replace(".jsonl", ".json")

    debug_jsonl_path = jsonl_path.replace(".jsonl", ".debug.jsonl") if args.debug else None
    debug_json_path  = jsonl_path.replace(".jsonl", ".debug.json")  if args.debug else None

    # ------------------------------------------------------------------
    # load already-processed items and reconstruct counters

    existing_results, n_done = _load_existing_jsonl(jsonl_path)

    pass_counts = {lid: 0 for lid in predicate_linker_ids}
    executable_count = 0
    timeout_count = 0
    runtime_agg = _new_runtime_agg(predicate_linker_ids)

    for r in existing_results:
        if r.get("executable"):
            executable_count += 1
            lid = r.get("winning_pass_linker", "")
            if lid in pass_counts:
                pass_counts[lid] += 1

        if r.get("timed_out"):
            timeout_count += 1

        if "item_runtime_sec" in r and r["item_runtime_sec"] is not None:
            _record_runtime(
                runtime_agg,
                r["item_runtime_sec"],
                r.get("winning_pass_linker"),
                timed_out=bool(r.get("timed_out")),
            )

    if n_done > 0:
        print(f"Resuming: {n_done}/{len(data)} items already processed, skipping ahead.")

    # Check if already fully done
    if n_done >= len(data):
        print("All items already processed. Finalising JSON output.")
        meta = _build_meta(args, entity_linker_ids, predicate_linker_ids, beam_limits,
                           k1_list, t1_list, k2_list, t2_list,
                           len(data), executable_count, timeout_count, pass_counts,
                           entity_linker_params, predicate_linker_params,
                           runtime_agg)
        out = _finalize_to_json(jsonl_path, meta)
        print(f"Finalised → {out}")
        if args.debug and debug_jsonl_path:
            debug_out = _finalize_to_json(debug_jsonl_path,
                                          {"meta": meta, "note": "debug"})
            print(f"Debug finalised → {debug_out}")
        return

    # ------------------------------------------------------------------

    print("\nResolving predictions...")
    print(f"  KB:               {args.kb}")
    print(f"  Entity linkers:   {entity_linker_ids}")
    print(f"  Entity params:    {entity_linker_params}")
    print(f"  Predicate passes: {predicate_linker_ids}")
    print(f"  Predicate params: {predicate_linker_params}")
    print(f"  Beam limits:      {[_get_pass_val(beam_limits, i) for i in range(n_passes)]}  (0 = unlimited)")
    print(f"  k1 per pass:      {[_get_pass_val(k1_list, i) for i in range(n_passes)]}")
    print(f"  t1 per pass:      {[_get_pass_val(t1_list, i) for i in range(n_passes)]}")
    print(f"  k2 per pass:      {[_get_pass_val(k2_list, i) for i in range(n_passes)]}")
    print(f"  t2 per pass:      {[_get_pass_val(t2_list, i) for i in range(n_passes)]}")
    print(f"  Item time limit:  {args.item_time_limit_sec if args.item_time_limit_sec is not None else 'none'}")
    print(f"  Endpoint:         {ENDPOINT_URL}")
    print(f"  Output (JSONL):   {jsonl_path}\n")

    for item_idx, item in enumerate(tqdm(data)):
        # Skip already-processed items
        if item_idx < n_done:
            continue

        question = item["question"]
        beams    = item["predict"]

        _log(f"item {item_idx} START | ID={item.get('ID')} | n_beams={len(beams)} | '{question[:60]}'")

        item_start = time.perf_counter()
        winning, all_passes = resolve_item(
            beams=beams,
            question=question,
            item=item,
            entity_linkers=entity_linkers,
            entity_linker_ids=entity_linker_ids,
            predicate_linkers=predicate_linkers,
            predicate_linker_ids=predicate_linker_ids,
            beam_limits=beam_limits,
            k1_list=k1_list,
            t1_list=t1_list,
            k2_list=k2_list,
            t2_list=t2_list,
            common_prefixes=common_prefixes,
            extract=extract,
            substitute=substitute,
            mode=args.mode,
            type_map=type_map,
            debug=args.debug,
            time_limit_sec=args.item_time_limit_sec,
        )
        item_runtime_sec = sum(pr.runtime_sec for pr in all_passes)
        _ = time.perf_counter() - item_start

        _log(f"item {item_idx} END   | found={winning.found} timed_out={winning.timed_out} beam_rank={winning.used_beam_rank} runtime={item_runtime_sec:.2f}s")

        if winning.found:
            executable_count += 1
            pass_counts[winning.pass_linker_id] += 1

        if winning.timed_out:
            timeout_count += 1

        if winning.found:
            executed_query = winning.executed_query
            executed_beam_rank = winning.used_beam_rank
            entity_map_used = winning.entity_map_used
            predicate_map_used = winning.predicate_map_used
            winning_pass_index = winning.pass_index
            winning_pass_linker = winning.pass_linker_id
            winning_entity_perm_idx = winning.entity_perm_idx
            winning_predicate_perm_idx = winning.predicate_perm_idx
        else:
            executed_query = None
            executed_beam_rank = None
            entity_map_used = None
            predicate_map_used = None
            winning_pass_index = None
            winning_pass_linker = None
            winning_entity_perm_idx = None
            winning_predicate_perm_idx = None

        _record_runtime(runtime_agg, item_runtime_sec, winning_pass_linker, timed_out=winning.timed_out)

        result = {
            **item,
            "executed_query":              executed_query,
            "executed_beam_rank":          executed_beam_rank,
            "entity_map_used":             entity_map_used,
            "predicate_map_used":          predicate_map_used,
            "winning_entity_perm_idx":     winning_entity_perm_idx,
            "winning_predicate_perm_idx":  winning_predicate_perm_idx,
            "executable":                  winning.found,
            "timed_out":                   winning.timed_out,
            "winning_pass_index":          winning_pass_index,
            "winning_pass_linker":         winning_pass_linker,
            "item_runtime_sec":            round(item_runtime_sec, 4),
            "pass_runtimes_sec": {
                pr.pass_linker_id: round(pr.runtime_sec, 4) for pr in all_passes
            },
        }
        _append_jsonl(jsonl_path, result)

        if args.debug:
            debug_entry = {
                "id":                  item.get("ID"),
                "question":            question,
                "gold_entity_map":     item.get("gold_entity_map", {}),
                "gold_relation_map":   item.get("gold_relation_map", {}),
                "gold_sexpr":          item.get("sexpr_with_labels", ""),
                "winning_pass_index":  winning_pass_index,
                "winning_pass_linker": winning_pass_linker,
                "timed_out":           winning.timed_out,
                "item_runtime_sec":    round(item_runtime_sec, 4),
                "passes": [
                    {
                        "pass_index":  pr.pass_index,
                        "pass_linker": pr.pass_linker_id,
                        "found":       pr.found,
                        "timed_out":   pr.timed_out,
                        "beam_rank":   pr.used_beam_rank,
                        "entity_perm_idx":    pr.entity_perm_idx,
                        "predicate_perm_idx": pr.predicate_perm_idx,
                        "runtime_sec": round(pr.runtime_sec, 4),
                        "beams":       sorted(pr.beam_debug, key=lambda x: x.get("rank", 0)),
                    }
                    for pr in all_passes
                ],
            }
            _append_jsonl(debug_jsonl_path, debug_entry)

    # ------------------------------------------------------------------
    # All items processed — finalise to JSON

    num_items = len(data)
    meta = _build_meta(args, entity_linker_ids, predicate_linker_ids, beam_limits,
                       k1_list, t1_list, k2_list, t2_list,
                       num_items, executable_count, timeout_count, pass_counts,
                       entity_linker_params, predicate_linker_params,
                       runtime_agg)

    out = _finalize_to_json(jsonl_path, meta)

    if args.debug and debug_jsonl_path:
        _finalize_to_json(debug_jsonl_path, {"meta": meta, "note": "debug"})
        print(f"Debug output → {debug_json_path}")

    print(f"\nDone. {executable_count}/{num_items} executable.")
    if args.item_time_limit_sec is not None:
        print(f"  Timed out (skipped): {timeout_count}/{num_items}")
    for lid, cnt in pass_counts.items():
        pct = round(cnt / num_items * 100, 1) if num_items else 0
        print(f"  Pass '{lid}': {cnt} items resolved ({pct}%)")
    rt_summary = meta["runtime"]
    print(f"Avg runtime/item: {rt_summary['avg_runtime_sec_per_item']}s")
    for key, bucket in rt_summary["runtime_by_resolution"].items():
        print(f"  resolved by '{key}': n={bucket['count']}  avg={bucket['avg_sec']}s")
    print(f"Saved to: {out}")


# ---------------------------------------------------------------------------
# Helpers used by main

def _build_meta(
    args,
    entity_linker_ids: list[str],
    predicate_linker_ids: list[str],
    beam_limits: list[int],
    k1_list: list[int],
    t1_list: list[float],
    k2_list: list[int],
    t2_list: list[float],
    num_items: int,
    executable_count: int,
    timeout_count: int,
    pass_counts: dict,
    entity_linker_params: dict,
    predicate_linker_params: dict,
    runtime_agg: dict,
) -> dict:
    n = len(predicate_linker_ids)
    return {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "dataset":          args.dataset,
        "split":            args.split,
        "model_id":         args.model_id,
        "kb":               args.kb,
        "mode":             args.mode,
        "entity_linkers":   entity_linker_ids,
        "entity_linker_params":    entity_linker_params,
        "predicate_linkers": predicate_linker_ids,
        "predicate_linker_params": predicate_linker_params,
        "beam_limits":      [_get_pass_val(beam_limits, i) for i in range(n)],
        "k1_per_pass":      [_get_pass_val(k1_list, i) for i in range(n)],
        "t1_per_pass":      [_get_pass_val(t1_list, i) for i in range(n)],
        "k2_per_pass":      [_get_pass_val(k2_list, i) for i in range(n)],
        "t2_per_pass":      [_get_pass_val(t2_list, i) for i in range(n)],
        "item_time_limit_sec": args.item_time_limit_sec,
        "endpoint":         ENDPOINT_URL,
        "data_dir":         args.data_dir,
        "note":             args.note,
        "num_items":        num_items,
        "num_executable":   executable_count,
        "executable_pct":   round(executable_count / num_items * 100, 2) if num_items else 0.0,
        "num_timed_out":    timeout_count,
        "timed_out_pct":    round(timeout_count / num_items * 100, 2) if num_items else 0.0,
        "pass_counts":      pass_counts,
        "runtime":          _runtime_summary(runtime_agg),
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("FATAL ERROR:", e)
        traceback.print_exc()
        raise