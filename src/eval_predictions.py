import os
import sys
import json
import argparse
import importlib.util
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.utils.sparql_exec import (
    init_uri_normaliser,
    normalise_gold_sparql,
    execute_sparql,
    bindings_to_rows,
    ensure_rows,
)

from src.utils.kb import load_kb_module

_DEFAULT_ENDPOINT = os.environ.get("ENDPOINT_URL", "https://query.wikidata.org/sparql")
_DATA_DIR = os.environ.get("DATA_DIR", "data")
_DEFAULT_LEDGER = "results/results.json"


# --------------------------------------------
# args

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate resolved KBQA predictions and record results."
    )

    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--split", required=True, help="Split: dev / test (/ train).")
    parser.add_argument("--mode", required=True, choices=["chatkbqa_webqsp", "chatkbqa_cwq", "jena", "sparql"],
                        help="Conversion mode used during resolution.")
    parser.add_argument("--model_id", required=True, help="Model identifier.")
    parser.add_argument("--entity_linkers", required=True,
                        help="Comma-seperated list of entity linkers used.")
    parser.add_argument("--predicate_linkers", required=True,
                        help="Comma-seperated list of predicate linkers used.")
    parser.add_argument("--endpoint", default=_DEFAULT_ENDPOINT,
                        help="SPARQL endpoint URL for execution.")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-query HTTP timeout in seconds.")

    parser.add_argument("--get-live-gold", action="store_true", default=False,
                        help="Execute the gold SPARQL live instead of using stored answers.")

    parser.add_argument("--ledger", default=_DEFAULT_LEDGER,
                        help="Path to the central results ledger JSON.")
    parser.add_argument("--note", default="",
                        help="Free-text note stored in the ledger entry for this run.")

    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of items evaluated (useful for debugging).")

    return parser.parse_args()


# --------------------------------------------
# Path helpers

def resolved_path(data_dir, dataset, model_id, entity_linkers, predicate_linkers, split, mode):
    return (Path(data_dir) / dataset / "predictions" / model_id / "resolved"
            / f"{'+'.join(entity_linkers.split(','))}+{'+'.join(predicate_linkers.split(','))}" / f"{dataset}_{split}.{mode}.json")


def evaluated_path(data_dir, dataset, model_id, entity_linkers, predicate_linkers, split, mode):
    return (Path(data_dir) / dataset / "predictions" / model_id / "evaluated"
            / f"{'+'.join(entity_linkers.split(','))}+{'+'.join(predicate_linkers.split(','))}" / f"{dataset}_{split}.{mode}.json")


# --------------------------------------------
# Metric

"""
From GRASP:
https://github.com/ad-freiburg/grasp/blob/7582dd1aeb3f70d4a952027cadada6901db41640/src/grasp/sparql/metrics.py
"""
def assignment_f1_score(
    pred: Iterable[Iterable],
    target: Iterable[Iterable],
) -> float:
    pred   = [Counter(p) for p in pred]
    target = [Counter(t) for t in target]
    scores = np.zeros((len(pred), len(target)), dtype=np.float32)
    for i, p_set in enumerate(pred):
        for j, t_set in enumerate(target):
            r = (p_set & t_set).total() / max(1, t_set.total())
            scores[i, j] = r
    rows, cols = linear_sum_assignment(scores, maximize=True)
    assert len(rows) == len(cols) == min(len(pred), len(target))
    assignment_scores = scores[rows, cols]
    tp = float(assignment_scores.sum())
    fn = float((1 - assignment_scores).sum()) + len(target) - len(rows)
    fp = len(pred) - len(rows)
    if tp <= 0.0:
        return 0.0
    prec = tp / (tp + fp)
    rec  = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def score(pred: list[list[str]], gold: list[list[str]]):
    """
    exact_match: pred and gold contain the same set of row
    assignment_f1: assignment F1 above
    hit1: f1 > 0 (min. one hit)
    """
    if not gold and not pred:
        return {"exact_match": 1, "assignment_f1": 1.0, "hit1": 1}

    if not gold or not pred:
        return {"exact_match": 0, "assignment_f1": 0.0, "hit1": 0}

    f1 = assignment_f1_score(pred, gold)

    # order should not matter for EM
    pred_bag = sorted(tuple(r) for r in pred)
    gold_bag = sorted(tuple(r) for r in gold)

    return {
        "exact_match":   int(pred_bag == gold_bag),
        "assignment_f1": round(f1, 4),
        "hit1":          int(f1 > 0.0),
    }


# --------------------------------------------
# Gold answer resolution

def get_gold_answers(item: dict, endpoint: str, timeout: int, get_live_gold: bool, common_prefixes) -> tuple[list[list[str]], str]: # answers, note
    raw_sparql = item.get("sparql", "")
    saved      = ensure_rows(item.get("answer", []))

    if not get_live_gold or not raw_sparql:
        return saved, "saved"

    normed, normed_err = normalise_gold_sparql(raw_sparql, common_prefixes)

    if not normed:
        return (saved, "saved_fallback") if saved else ([], "empty")

    raw = execute_sparql(normed, endpoint, timeout)

    if raw is not None:
        rows = bindings_to_rows(raw)
        if rows:
            return rows, "live"

    # Live query returned nothing
    return (saved, "saved_fallback") if saved else ([], "empty")


# --------------------------------------------
# Ledger helpers

def load_ledger(path: str) -> list[dict]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def save_ledger(ledger: list[dict], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------
# Main

def main():
    args = parse_args()

    res_path = resolved_path(_DATA_DIR, args.dataset, args.model_id, args.entity_linkers, args.predicate_linkers, args.split, args.mode)
    eval_out = evaluated_path(_DATA_DIR, args.dataset, args.model_id, args.entity_linkers, args.predicate_linkers, args.split, args.mode)

    if not res_path.exists():
        raise FileNotFoundError(f"Resolved file not found: {res_path}")

    print(f"Loading resolved predictions: {res_path}")
    raw = json.loads(res_path.read_text(encoding="utf-8"))

    file_meta: dict = raw["meta"]
    items: list[dict] = raw["items"]

    kb_name = file_meta.get("kb", "freebase")
    kb_module = load_kb_module(kb_name)
    common_prefixes = getattr(kb_module, "COMMON_PREFIXES", {})
    init_uri_normaliser(kb_module)
    print(f"Loaded KB module '{kb_name}' ({len(common_prefixes)} prefixes)")

    print(f"Loaded {len(items)} items")
    predicate_linkers_str = "+".join(file_meta.get("predicate_linkers") or [])
    print(f"  Resolved with: {file_meta.get('model_id')} / {file_meta.get('mode')} mode "
          f"/ {file_meta.get('entity_linker')}+{predicate_linkers_str}")

    gold_source_label = "live SPARQL execution" if args.get_live_gold else "saved"
    print(f"  Gold answers:  {gold_source_label}")

    if args.max_samples:
        items = items[: args.max_samples]
        print(f"Capped to {len(items)} items")

    print()
    print(f"Evaluating against: {args.endpoint}")
    print(f"Evaluated file:     {eval_out}")
    print(f"Ledger:             {args.ledger}\n")

    evaluated_items = []
    totals: dict[str, float] = defaultdict(float)
    n_executable = 0
    n_with_gold = 0
    n_empty_pred = 0
    n_exec_error = 0
    gold_source_counts: dict[str, int] = defaultdict(int)

    for item in tqdm(items):
        item_id = item.get("ID", item.get("id", ""))
        question = item.get("question", "")
        query = item.get("executed_query")

        # get gold answers + source
        gold_rows, gold_src = get_gold_answers(
            item, args.endpoint, args.timeout, args.get_live_gold, common_prefixes
        )
        gold_source_counts[gold_src] += 1
        if gold_rows:
            n_with_gold += 1

        # pred answers
        pred_rows: list[list[str]] = []
        exec_status = "no_query"

        if query and item.get("executable", False):
            n_executable += 1
            bindings = execute_sparql(query, args.endpoint, args.timeout)

            if bindings is None:
                exec_status  = "error"
                n_exec_error += 1
            else:
                pred_rows = bindings_to_rows(bindings)
                if pred_rows:
                    exec_status = "ok"
                else:
                    exec_status  = "empty"
                    n_empty_pred += 1

        # scoring
        item_scores = score(pred_rows, gold_rows)
        for k, v in item_scores.items():
            totals[k] += v

        evaluated_items.append({
            "ID":               item_id,
            "question":         question,
            "gold_sparql":      item.get("sparql", ""),
            "gold_entity_map":  item.get("gold_entity_map",   {}),
            "gold_relation_map": item.get("gold_relation_map", {}),
            "gold_answers":     gold_rows,
            "gold_answer_source": gold_src,
            "pred_sparql":      query,
            "pred_entity_map":  item.get("entity_map_used",   {}),
            "pred_relation_map": item.get("predicate_map_used", {}),
            "pred_answers":     pred_rows,
            "executed_beam_rank": item.get("executed_beam_rank"),
            "executable":       item.get("executable", False),
            "exec_status":      exec_status,
            **item_scores,
        })

    n = len(items)

    def pct(x): return round(x / n * 100, 2) if n else 0.0

    aggregate = {
        "num_items":       n,
        "num_executable":  n_executable,
        "executable_pct":  pct(n_executable),
        "num_exec_error":  n_exec_error,
        "num_empty_pred":  n_empty_pred,
        "num_with_gold":   n_with_gold,
        "exact_match":     round(totals["exact_match"]    / n, 4) if n else 0.0,
        "assignment_f1":   round(totals["assignment_f1"]  / n, 4) if n else 0.0,
        "hit1":            round(totals["hit1"]           / n, 4) if n else 0.0,
    }

    print("\n" + "=" * 50)
    print(f"  Dataset:        {file_meta.get('dataset')} / {file_meta.get('split')}")
    print(f"  Model:          {file_meta.get('model_id')}")
    print(f"  Mode:           {file_meta.get('mode')}")
    print(f"  KB:             {kb_name}")
    print(f"  Entity Linkers: {args.entity_linkers.split(',')}")
    print(f"  Predicate Linkers: {args.predicate_linkers.split(',')}")
    print(f"  Eval endpoint:  {args.endpoint}")
    print(f"  Gold source:    {gold_source_label}")
    if args.get_live_gold:
        print(f"    live={gold_source_counts.get('live', 0)}  "
              f"saved_fallback={gold_source_counts.get('saved_fallback', 0)}  "
              f"empty={gold_source_counts.get('empty', 0)}")
    print(f"  Items:          {n}")
    print(f"  Executable:     {n_executable} ({aggregate['executable_pct']}%)")
    print(f"  Exec errors:    {n_exec_error}")
    print(f"  Empty results:  {n_empty_pred}")
    print("-" * 50)
    print(f"  Exact Match:    {aggregate['exact_match']:.4f}")
    print(f"  Assignment F1:  {aggregate['assignment_f1']:.4f}")
    print(f"  Hit@1:          {aggregate['hit1']:.4f}")
    print("=" * 50 + "\n")

    # write eval file
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    eval_payload = {
        "meta": {
            **file_meta,
            "eval_timestamp":     datetime.now(timezone.utc).isoformat(),
            "eval_endpoint":      args.endpoint,
            "gold_answer_source": "live" if args.get_live_gold else "saved",
            "eval_note":          args.note,
            **aggregate,
        },
        "items": evaluated_items,
    }
    eval_out.write_text(
        json.dumps(eval_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Evaluated file  → {eval_out}")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolved_file": str(res_path.resolve()),
        "evaluated_file": str(eval_out.resolve()),
        "eval_endpoint": args.endpoint,
        "gold_answer_source": "live" if args.get_live_gold else "saved",
        "resolve_note": file_meta.get("note", ""),
        "eval_note": args.note,
        "dataset": file_meta.get("dataset"),
        "split": file_meta.get("split"),
        "model_id": file_meta.get("model_id"),
        "kb": file_meta.get("kb"),
        "mode": file_meta.get("mode"),
        "entity_linker": file_meta.get("entity_linkers"),
        "entity_linker_params": file_meta.get("entity_linker_params"),
        "predicate_linkers": file_meta.get("predicate_linkers"),
        "predicate_linker_params": file_meta.get("predicate_linker_params"),
        "beam_limits": file_meta.get("beam_limits"),
        "k1_per_pass": file_meta.get("k1_per_pass"),
        "t1_per_pass": file_meta.get("t1_per_pass"),
        "k2_per_pass": file_meta.get("k2_per_pass"),
        "t2_per_pass": file_meta.get("t2_per_pass"),
        "pass_counts": file_meta.get("pass_counts"),
        "resolve_endpoint": file_meta.get("endpoint"),
        "resolve_timestamp": file_meta.get("timestamp"),
        **aggregate,
    }

    ledger = load_ledger(args.ledger)
    ledger.append(entry)
    save_ledger(ledger, args.ledger)
    print(f"Ledger updated  → {args.ledger}  ({len(ledger)} total runs)")


if __name__ == "__main__":
    main()