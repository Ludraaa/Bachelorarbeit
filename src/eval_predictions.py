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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# F1 budgets for the hyperparameter-sensitivity analysis, expressed as
# absolute points on the 0-1 assignment_f1 scale (0.01% -> 0.0001, etc).
_F1_BUDGETS = {
    "0.01pct": 0.0001,
    "0.1pct":  0.001,
    "1pct":    0.01,
}

# Caps (beam_limit, k1, k2) are never allowed to drop below this. A cap of 0
# is ambiguous (it would mean "keep nothing", but for beam_limit specifically
# downstream code used 0 to mean "unlimited") so we floor every cap at 1,
# meaning at minimum the single best (rank/perm-idx 0) candidate is always
# kept.
_MIN_CAP = 1

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.constrained_layout.use": True,
})


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

    parser.add_argument("--skip_analysis", action="store_true", default=False,
                        help="Skip the distribution/hyperparameter-sensitivity analysis and plots.")

    return parser.parse_args()


# --------------------------------------------
# Path helpers

def resolved_path(data_dir, dataset, model_id, entity_linkers, predicate_linkers, split, mode):
    return (Path(data_dir) / dataset / "predictions" / model_id / "resolved"
            / f"{'+'.join(entity_linkers.split(','))}+{'+'.join(predicate_linkers.split(','))}" / f"{dataset}_{split}.{mode}.json")


def evaluated_path(data_dir, dataset, model_id, entity_linkers, predicate_linkers, split, mode):
    return (Path(data_dir) / dataset / "predictions" / model_id / "evaluated"
            / f"{'+'.join(entity_linkers.split(','))}+{'+'.join(predicate_linkers.split(','))}" / f"{dataset}_{split}.{mode}.json")


def analysis_json_path(eval_out: Path) -> Path:
    return eval_out.with_suffix("").with_suffix(".analysis.json")


def analysis_plots_dir(eval_out: Path) -> Path:
    return eval_out.parent / f"{eval_out.stem}_plots"


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
# distribution analysis


def _idx_stats(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.array(values, dtype=float)
    return {
        "count":  len(values),
        "min":    int(arr.min()),
        "max":    int(arr.max()),
        "mean":   round(float(arr.mean()), 3),
        "median": float(np.median(arr)),
        "p90":    float(np.percentile(arr, 90)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
    }


def _float_stats(values: list[float]) -> dict:
    """Same idea as _idx_stats but for continuous scores (e.g. per-item
    assignment_f1) where int-casting min/max would be wrong."""
    if not values:
        return {"count": 0}
    arr = np.array(values, dtype=float)
    return {
        "count":  len(values),
        "mean":   round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "std":    round(float(arr.std()), 4),
        "min":    round(float(arr.min()), 4),
        "max":    round(float(arr.max()), 4),
    }


def build_distribution_analysis(evaluated_items: list[dict], predicate_linkers: list[str]) -> dict:
    n = len(evaluated_items)

    winning_pass_counter = Counter(
        it.get("winning_pass_linker") or "_unresolved" for it in evaluated_items
    )
    winning_pass = {
        lid: {"count": cnt, "pct": round(cnt / n * 100, 2) if n else 0.0}
        for lid, cnt in winning_pass_counter.items()
    }

    beam_by_pass = {}
    entity_perm_by_pass = {}
    predicate_perm_by_pass = {}

    for lid in predicate_linkers:
        items_for_pass = [it for it in evaluated_items if it.get("winning_pass_linker") == lid]
        beam_by_pass[lid] = _idx_stats(
            [it["executed_beam_rank"] for it in items_for_pass if it.get("executed_beam_rank") is not None]
        )
        entity_perm_by_pass[lid] = _idx_stats(
            [it["winning_entity_perm_idx"] for it in items_for_pass if it.get("winning_entity_perm_idx") is not None]
        )
        predicate_perm_by_pass[lid] = _idx_stats(
            [it["winning_predicate_perm_idx"] for it in items_for_pass if it.get("winning_predicate_perm_idx") is not None]
        )

    return {
        "winning_pass": winning_pass,
        "winning_beam_rank_by_pass": beam_by_pass,
        "winning_entity_perm_by_pass": entity_perm_by_pass,
        "winning_predicate_perm_by_pass": predicate_perm_by_pass,
    }


def _exec_status_breakdown(items_for_pass: list[dict]) -> dict:
    n = len(items_for_pass)
    counter = Counter(it.get("exec_status", "no_query") for it in items_for_pass)
    return {
        status: {"count": c, "pct": round(c / n * 100, 2) if n else 0.0}
        for status, c in counter.items()
    }


def build_per_linker_performance(evaluated_items: list[dict], predicate_linkers: list[str]) -> dict:
    """Executability and answer-quality (EM / assignment_f1 / Hit@1) broken
    down by which predicate-linker pass produced the winning query for each
    item. This is the eval-time counterpart to the resolve-time
    `runtime_by_resolution` breakdown already stored in the resolved-file
    meta: runtime tells you how expensive a pass is, this tells you whether
    it was actually worth it — a pass can be cheap but only ever contribute
    low-quality answers, or expensive but responsible for most of the
    correct ones. Items with no winning pass (unresolved) are reported
    under "_unresolved" for comparison.
    """
    n_total = len(evaluated_items)
    pass_labels = list(predicate_linkers) + ["_unresolved"]

    out = {}
    for lid in pass_labels:
        if lid == "_unresolved":
            items_for_pass = [it for it in evaluated_items if not it.get("winning_pass_linker")]
        else:
            items_for_pass = [it for it in evaluated_items if it.get("winning_pass_linker") == lid]

        n = len(items_for_pass)
        if n == 0:
            continue

        n_executable = sum(1 for it in items_for_pass if it.get("executable"))
        n_ok = sum(1 for it in items_for_pass if it.get("exec_status") == "ok")

        out[lid] = {
            "count":               n,
            "pct_of_total":        round(n / n_total * 100, 2) if n_total else 0.0,
            "executable_count":    n_executable,
            "executable_pct":      round(n_executable / n * 100, 2),
            "exec_ok_count":       n_ok,
            "exec_ok_pct":         round(n_ok / n * 100, 2),
            "exec_status_breakdown": _exec_status_breakdown(items_for_pass),
            "exact_match_rate":    round(sum(it["exact_match"] for it in items_for_pass) / n, 4),
            "hit1_rate":           round(sum(it["hit1"] for it in items_for_pass) / n, 4),
            "assignment_f1":       _float_stats([it["assignment_f1"] for it in items_for_pass]),
        }
    return out


# --------------------------------------------
# hyperparameter


def _grouped_losses(items_for_pass: list[dict], idx_key: str) -> dict[int, list[float]]:
    groups: dict[int, list[float]] = defaultdict(list)
    for it in items_for_pass:
        idx = it.get(idx_key)
        if idx is None or idx < _MIN_CAP:
            continue
        groups[idx].append(it["assignment_f1"])
    return groups


def _param_sensitivity(items_for_pass: list[dict], idx_key: str, original_cap: float, n_total: int) -> dict:
    groups = _grouped_losses(items_for_pass, idx_key)
    orig_cap = max(original_cap, _MIN_CAP)

    if not groups:
        return {
            budget_name: {
                "original_cap": orig_cap,
                "new_cap": orig_cap,
                "items_dropped": 0,
                "f1_loss": 0.0,
            }
            for budget_name in _F1_BUDGETS
        }

    distinct_idxs_desc = sorted(groups.keys(), reverse=True)

    results = {}
    for budget_name, budget in _F1_BUDGETS.items():
        cum_loss = 0.0
        dropped_items = 0
        cap = orig_cap
        for idx_val in distinct_idxs_desc:
            group_loss = sum(groups[idx_val]) / n_total
            if cum_loss + group_loss <= budget:
                cum_loss += group_loss
                dropped_items += len(groups[idx_val])
                cap = idx_val
            else:
                break
        results[budget_name] = {
            "original_cap": orig_cap,
            "new_cap": max(cap, _MIN_CAP),
            "items_dropped": dropped_items,
            "f1_loss": round(cum_loss, 6),
        }
    return results


def _full_sensitivity_curve(items_for_pass: list[dict], idx_key: str, original_cap: float, n_total: int) -> dict:
    """Cumulative F1 loss at every distinct cap value that actually occurs
    in the data, not just the three budget-selected caps. Gives a full
    step-function curve (for plotting) instead of 3 sparse points. Caps
    never go below _MIN_CAP, matching _param_sensitivity."""
    groups = _grouped_losses(items_for_pass, idx_key)
    orig_cap = max(original_cap, _MIN_CAP)

    if not groups:
        return {"caps": [orig_cap], "cum_loss_pct": [0.0]}

    distinct_idxs_desc = sorted(groups.keys(), reverse=True)
    caps = [orig_cap]
    cum_loss_pct = [0.0]
    cum_loss = 0.0
    for idx_val in distinct_idxs_desc:
        cum_loss += sum(groups[idx_val]) / n_total
        caps.append(idx_val)
        cum_loss_pct.append(round(cum_loss * 100, 4))

    return {"caps": caps, "cum_loss_pct": cum_loss_pct}


def _combined_effect(items_for_pass: list[dict], new_beam_cap: float, new_k1: float, new_k2: float, n_total: int) -> dict:
    new_beam_cap = max(new_beam_cap, _MIN_CAP)
    new_k1 = max(new_k1, _MIN_CAP)
    new_k2 = max(new_k2, _MIN_CAP)
    loss = 0.0
    dropped = 0
    for it in items_for_pass:
        beam_ok = it.get("executed_beam_rank", 0) < new_beam_cap
        k1_ok = it.get("winning_entity_perm_idx", 0) < new_k1
        k2_ok = it.get("winning_predicate_perm_idx", 0) < new_k2
        if not (beam_ok and k1_ok and k2_ok):
            loss += it["assignment_f1"]
            dropped += 1
    return {"items_dropped": dropped, "f1_loss": round(loss / n_total, 6) if n_total else 0.0}


def build_hyperparam_analysis(evaluated_items: list[dict], file_meta: dict, n_total: int) -> dict:
    predicate_linkers = file_meta.get("predicate_linkers") or []
    beam_limits = file_meta.get("beam_limits") or []
    k1_list = file_meta.get("k1_per_pass") or []
    k2_list = file_meta.get("k2_per_pass") or []

    out = {}
    for pass_idx, lid in enumerate(predicate_linkers):
        items_for_pass = [it for it in evaluated_items if it.get("winning_pass_linker") == lid]
        if not items_for_pass:
            continue

        orig_beam_limit = max(beam_limits[pass_idx], _MIN_CAP)
        orig_k1 = max(k1_list[pass_idx], _MIN_CAP)
        orig_k2 = max(k2_list[pass_idx], _MIN_CAP)

        total_pass_runtime = sum(
            (it.get("pass_runtimes_sec") or {}).get(lid, 0.0) for it in evaluated_items
        )

        beam_sens = _param_sensitivity(items_for_pass, "executed_beam_rank", orig_beam_limit, n_total)
        k1_sens = _param_sensitivity(items_for_pass, "winning_entity_perm_idx", orig_k1, n_total)
        k2_sens = _param_sensitivity(items_for_pass, "winning_predicate_perm_idx", orig_k2, n_total)

        beam_curve = _full_sensitivity_curve(items_for_pass, "executed_beam_rank", orig_beam_limit, n_total)
        k1_curve = _full_sensitivity_curve(items_for_pass, "winning_entity_perm_idx", orig_k1, n_total)
        k2_curve = _full_sensitivity_curve(items_for_pass, "winning_predicate_perm_idx", orig_k2, n_total)

        combined = {}
        for budget_name in _F1_BUDGETS:
            new_beam_cap = max(beam_sens.get(budget_name, {}).get("new_cap", orig_beam_limit), _MIN_CAP)
            new_k1 = max(k1_sens.get(budget_name, {}).get("new_cap", orig_k1), _MIN_CAP)
            new_k2 = max(k2_sens.get(budget_name, {}).get("new_cap", orig_k2), _MIN_CAP)

            effect = _combined_effect(items_for_pass, new_beam_cap, new_k1, new_k2, n_total)

            scale = (
                min(1.0, new_beam_cap / orig_beam_limit)
                * min(1.0, new_k1 / orig_k1)
                * min(1.0, new_k2 / orig_k2)
            )
            estimated_new_runtime = total_pass_runtime * scale
            time_saved = total_pass_runtime - estimated_new_runtime

            combined[budget_name] = {
                "new_beam_limit": new_beam_cap,
                "new_k1": new_k1,
                "new_k2": new_k2,
                "actual_items_dropped": effect["items_dropped"],
                "actual_f1_loss": effect["f1_loss"],
                "estimated_time_saved_sec": round(time_saved, 2),
                "estimated_time_saved_pct": round(time_saved / total_pass_runtime * 100, 2) if total_pass_runtime else 0.0,
            }

        out[lid] = {
            "original_beam_limit": orig_beam_limit,
            "original_k1": orig_k1,
            "original_k2": orig_k2,
            "total_pass_runtime_sec": round(total_pass_runtime, 2),
            "beam_limit_sensitivity": beam_sens,
            "k1_sensitivity": k1_sens,
            "k2_sensitivity": k2_sens,
            "beam_limit_curve": beam_curve,
            "k1_curve": k1_curve,
            "k2_curve": k2_curve,
            "combined_estimate": combined,
        }

    return out


# --------------------------------------------
# plots

def _hist_ax(ax, values: list[int], title: str, xlabel: str):
    if not values:
        ax.set_title(f"{title} (no data)")
        return
    max_v = max(values)
    bins = min(max_v + 2, 40)
    ax.hist(values, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# items (log scale)")


def generate_plots(evaluated_items: list[dict], file_meta: dict, distributions: dict,
                    hyperparam: dict, n_total: int, plots_dir: Path) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    predicate_linkers = file_meta.get("predicate_linkers") or []

    # 1. Winning-pass distribution
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = list(distributions["winning_pass"].keys())
    counts = [distributions["winning_pass"][l]["count"] for l in labels]
    colors = ["#DD8452" if l == "_unresolved" else "#4C72B0" for l in labels]
    ax.bar(labels, counts, color=colors)
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# items")
    ax.set_title(f"Winning pass distribution (n={n_total})")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    path = plots_dir / "winning_pass_distribution.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    saved.append(path)

    # 2. Per-pass histograms: beam rank / entity perm idx / predicate perm idx
    for lid in predicate_linkers:
        items_for_pass = [it for it in evaluated_items if it.get("winning_pass_linker") == lid]
        if not items_for_pass:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        _hist_ax(axes[0], [it["executed_beam_rank"] for it in items_for_pass if it.get("executed_beam_rank") is not None],
                 "Winning beam rank", "beam rank")
        _hist_ax(axes[1], [it["winning_entity_perm_idx"] for it in items_for_pass if it.get("winning_entity_perm_idx") is not None],
                 "Winning entity permutation idx", "entity perm idx")
        _hist_ax(axes[2], [it["winning_predicate_perm_idx"] for it in items_for_pass if it.get("winning_predicate_perm_idx") is not None],
                 "Winning predicate permutation idx", "predicate perm idx")
        fig.suptitle(f"Pass '{lid}' (n={len(items_for_pass)})")
        path = plots_dir / f"winning_index_distributions_{lid.replace('.', '_')}.pdf"
        fig.savefig(path)
        fig.savefig(path.with_suffix(".png"))
        plt.close(fig)
        saved.append(path)

    # 3. Full F1-vs-cap sensitivity curves per pass/param, with the three
    #    budget picks annotated on top of the continuous curve.
    for lid, data in hyperparam.items():
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        param_specs = [
            ("beam_limit_sensitivity", "beam_limit_curve", axes[0], "beam_limit"),
            ("k1_sensitivity", "k1_curve", axes[1], "k1"),
            ("k2_sensitivity", "k2_curve", axes[2], "k2"),
        ]
        budget_order = ["0.01pct", "0.1pct", "1pct"]
        for sens_key, curve_key, ax, pname in param_specs:
            curve = data.get(curve_key) or {}
            sens = data.get(sens_key) or {}
            if not curve.get("caps"):
                ax.set_title(f"{pname} (no data)")
                continue

            ax.plot(curve["caps"], curve["cum_loss_pct"], color="#4C72B0", linewidth=1.2)

            for b in budget_order:
                if b not in sens:
                    continue
                cap = sens[b]["new_cap"]
                loss_pct = sens[b]["f1_loss"] * 100
                ax.scatter([cap], [loss_pct], color="#DD8452", zorder=5, s=25)
                ax.annotate(b, (cap, loss_pct), textcoords="offset points", xytext=(4, 4), fontsize=7)

            ax.set_xlabel(f"{pname} cap")
            ax.set_ylabel("cumulative F1 loss (%)")
            ax.set_title(pname)
            ax.invert_xaxis()
        fig.suptitle(f"F1 cost of trimming search caps — pass '{lid}' (min cap = {_MIN_CAP})")
        path = plots_dir / f"sensitivity_{lid.replace('.', '_')}.pdf"
        fig.savefig(path)
        fig.savefig(path.with_suffix(".png"))
        plt.close(fig)
        saved.append(path)

    # 4. Estimated time savings per pass at each budget
    if hyperparam:
        budget_order = ["0.01pct", "0.1pct", "1pct"]
        passes = list(hyperparam.keys())
        fig, ax = plt.subplots(figsize=(6, 3.8))
        width = 0.8 / len(budget_order)
        x = np.arange(len(passes))
        for i, b in enumerate(budget_order):
            vals = [hyperparam[p]["combined_estimate"].get(b, {}).get("estimated_time_saved_pct", 0.0) for p in passes]
            ax.bar(x + i * width, vals, width, label=b)
        ax.set_xticks(x + width, passes, rotation=15, ha="right")
        ax.set_ylabel("estimated time saved (%)")
        ax.set_title("Estimated per-pass runtime savings vs. F1 budget")
        ax.legend(title="F1 budget")
        path = plots_dir / "estimated_time_savings.pdf"
        fig.savefig(path)
        fig.savefig(path.with_suffix(".png"))
        plt.close(fig)
        saved.append(path)

    return saved


def generate_per_linker_plots(evaluated_items: list[dict], predicate_linkers: list[str],
                               per_linker_performance: dict, plots_dir: Path) -> list[Path]:
    """Plots for build_per_linker_performance: answer quality broken down by
    winning predicate-linker pass. Meant to sit next to the existing
    runtime-by-pass numbers in the resolve-file meta so a pass's cost and
    its actual payoff can be read side by side."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    pass_labels = [l for l in list(predicate_linkers) + ["_unresolved"] if l in per_linker_performance]
    if not pass_labels:
        return saved

    colors = ["#DD8452" if l == "_unresolved" else "#4C72B0" for l in pass_labels]
    x = np.arange(len(pass_labels))

    # 1. Answer quality per pass: EM / mean assignment F1 / Hit@1
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    width = 0.25
    em = [per_linker_performance[l]["exact_match_rate"] for l in pass_labels]
    f1 = [per_linker_performance[l]["assignment_f1"]["mean"] for l in pass_labels]
    h1 = [per_linker_performance[l]["hit1_rate"] for l in pass_labels]
    ax.bar(x - width, em, width, label="exact match", color="#4C72B0")
    ax.bar(x, f1, width, label="assignment F1 (mean)", color="#DD8452")
    ax.bar(x + width, h1, width, label="hit@1", color="#55A868")
    ax.set_xticks(x, pass_labels)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("Answer quality by winning pass")
    ax.legend()
    path = plots_dir / "quality_by_pass.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    saved.append(path)

    # 2. Per-item F1 distribution per pass (boxplot) — shows spread, not just
    #    the mean; a pass can have the same mean F1 as another but be far
    #    more bimodal (mostly-right-or-totally-wrong vs. consistently-okay).
    f1_by_pass = []
    for l in pass_labels:
        if l == "_unresolved":
            items_for_pass = [it for it in evaluated_items if not it.get("winning_pass_linker")]
        else:
            items_for_pass = [it for it in evaluated_items if it.get("winning_pass_linker") == l]
        f1_by_pass.append([it["assignment_f1"] for it in items_for_pass])

    fig, ax = plt.subplots(figsize=(6, 3.8))
    bp = ax.boxplot(f1_by_pass, labels=pass_labels, showmeans=True, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_ylabel("assignment F1")
    ax.set_title("Per-item F1 distribution by winning pass")
    path = plots_dir / "f1_distribution_by_pass.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    saved.append(path)

    # 3. Share of total items vs. share of total F1 contributed, per pass.
    #    Makes "worthwhileness" visible directly: a pass that eats a large
    #    slice of the item pie but only a thin slice of the F1 pie is a
    #    prime candidate for cutting/deprioritizing.
    f1_sums = {
        l: per_linker_performance[l]["assignment_f1"]["mean"] * per_linker_performance[l]["count"]
        for l in pass_labels
    }
    total_f1 = sum(f1_sums.values())
    item_share = [per_linker_performance[l]["count"] for l in pass_labels]
    total_items = sum(item_share)
    item_share_pct = [round(v / total_items * 100, 2) if total_items else 0.0 for v in item_share]
    f1_share_pct = [round(f1_sums[l] / total_f1 * 100, 2) if total_f1 else 0.0 for l in pass_labels]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    width = 0.35
    ax.bar(x - width / 2, item_share_pct, width, label="share of all items", color="#4C72B0")
    ax.bar(x + width / 2, f1_share_pct, width, label="share of total assignment F1", color="#55A868")
    ax.set_xticks(x, pass_labels)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_ylabel("% share")
    ax.set_title("Item share vs. F1 contribution share by pass")
    ax.legend()
    path = plots_dir / "item_vs_f1_share_by_pass.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    saved.append(path)

    return saved


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
            "winning_pass_index":         item.get("winning_pass_index"),
            "winning_pass_linker":        item.get("winning_pass_linker"),
            "winning_entity_perm_idx":    item.get("winning_entity_perm_idx"),
            "winning_predicate_perm_idx": item.get("winning_predicate_perm_idx"),
            "item_runtime_sec":           item.get("item_runtime_sec"),
            "pass_runtimes_sec":          item.get("pass_runtimes_sec"),
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

    # ------------------------------------------------------------------
    # distribution + hyperparameter + per-linker analysis

    if args.skip_analysis:
        return

    print("\nRunning analysis (winning-pass/permutation distributions, "
          "per-linker executability & F1 impact, hyperparameter sensitivity)...")

    predicate_linkers = file_meta.get("predicate_linkers") or []
    distributions = build_distribution_analysis(evaluated_items, predicate_linkers)
    per_linker_performance = build_per_linker_performance(evaluated_items, predicate_linkers)
    hyperparam = build_hyperparam_analysis(evaluated_items, file_meta, n)

    print("\nPer-pass executability & quality:")
    for lid, d in per_linker_performance.items():
        print(f"  {lid:35s} n={d['count']:5d} ({d['pct_of_total']:5.2f}%)  "
              f"exec={d['executable_pct']:6.2f}%  EM={d['exact_match_rate']:.4f}  "
              f"F1={d['assignment_f1']['mean']:.4f}  Hit@1={d['hit1_rate']:.4f}")

    analysis_payload = {
        "meta": {
            "source_evaluated_file": str(eval_out.resolve()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "num_items": n,
            "aggregate_assignment_f1": aggregate["assignment_f1"],
            "f1_budgets": _F1_BUDGETS,
            "min_cap": _MIN_CAP,
            "note": (
                "hyperparameter_sensitivity assumes an item dropped by a "
                "cap reduction becomes permanently unresolved (worst case). "
                f"beam_limit/k1/k2 caps are floored at {_MIN_CAP} (the top "
                "candidate per pass is never droppable). *_curve fields give "
                "the full cumulative-F1-loss-vs-cap curve, not just the "
                "three budget-selected points. per_linker_performance is "
                "the eval-time counterpart to the resolve-file's "
                "runtime_by_resolution: it reports executability and "
                "answer-quality (EM/assignment_f1/Hit@1), grouped by which "
                "predicate-linker pass actually won each item, so pass cost "
                "(runtime) and pass payoff (quality) can be compared "
                "directly."
            ),
        },
        "distributions": distributions,
        "per_linker_performance": per_linker_performance,
        "hyperparameter_sensitivity": hyperparam,
    }

    a_path = analysis_json_path(eval_out)
    a_path.write_text(json.dumps(analysis_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAnalysis file   → {a_path}")

    plots_dir = analysis_plots_dir(eval_out)
    saved_plots = generate_plots(evaluated_items, file_meta, distributions, hyperparam, n, plots_dir)
    saved_plots += generate_per_linker_plots(evaluated_items, predicate_linkers, per_linker_performance, plots_dir)
    print(f"Analysis plots  → {plots_dir}  ({len(saved_plots)} figures, PDF + PNG)")


if __name__ == "__main__":
    main()