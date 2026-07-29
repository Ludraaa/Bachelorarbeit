import os
import re
import sys
import json
import argparse
import importlib.util
import statistics
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

import torch
import yaml
from llamafactory.chat import ChatModel

from sexpr.jena_interface import algebra_to_sparql, sparql_to_algebra, fix_sparql_for_jena


# ---------------------------------------------------------------------------
# KB module loader

def load_kb_module(kb_name: str):
    path = Path(f"src/kb/{kb_name}.py")
    if not path.is_file():
        print(f"Error: KB module not found: {path}", file=sys.stderr)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("kb_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="LlamaFactory inference yaml")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--mode", type=str, default="simple",
                        choices=["simple", "jena", "sparql"])
    parser.add_argument("--kb", type=str, default="freebase",
                        help="KB module name under src/kb/")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset

def load_dataset(dataset, split, mode, data_dir):
    path = os.path.join(
        data_dir, dataset, "generation", "merged",
        f"{dataset}_{split}.{mode}.json"
    )
    print(f"Loading dataset: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} examples")
    return data


# ---------------------------------------------------------------------------
# Generation

INSTRUCTION = (
    "Generate a Logical Form query that retrieves the information corresponding to the given question."
)

def build_question(raw_question: str) -> str:
    return f"{INSTRUCTION}\n\nQuestion: {{ {raw_question} }}"


def generate_beams(engine, messages: list[dict], num_beams: int, max_new_tokens: int) -> list[str]:
    tok = engine.tokenizer
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(engine.model.device)
    input_len = inputs["input_ids"].shape[1]
    
    engine.model.generation_config.cache_implementation = None # maybe this helps?

    with torch.no_grad():
        outputs = engine.model.generate(
            **inputs,
            custom_generate="transformers-community/group-beam-search",
            num_beams=num_beams,
            num_beam_groups=num_beams,
            diversity_penalty=1.0,
            num_return_sequences=num_beams,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=max_new_tokens,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            trust_remote_code=True,
        )

    decoded = [
        tok.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]

    # Deduplicate
    seen = set()
    unique = []
    for p in decoded:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# Parseability checks

def _is_parseable_jena(beam: str) -> bool:
    try:
        result = algebra_to_sparql(beam)
        return result is not None
    except Exception:
        return False


def _is_parseable_sparql(beam: str, common_prefixes: dict) -> bool:
    try:
        normalised = fix_sparql_for_jena(beam, common_prefixes)
        result = sparql_to_algebra(normalised)
        return result is not None
    except Exception:
        return False


def is_parseable(beam: str, mode: str, common_prefixes: dict) -> bool:
    if mode == "jena":
        return _is_parseable_jena(beam)
    if mode == "sparql":
        return _is_parseable_sparql(beam, common_prefixes)
    return False


# ---------------------------------------------------------------------------
# Per-item statistics helpers

def _gold_rank(predictions: list[str], gold: str) -> int | None:
    """Return the rank (0-indexed) of the gold sexpr in predictions, or None."""
    gold_lower = gold.strip().lower()
    for rank, p in enumerate(predictions):
        if p.strip().lower() == gold_lower:
            return rank
    return None


def _count_parseable(predictions: list[str], mode: str, common_prefixes: dict) -> int:
    return sum(1 for p in predictions if is_parseable(p, mode, common_prefixes))


# ---------------------------------------------------------------------------
# Metadata builder

def _build_meta(
    args,
    model_id: str,
    num_items: int,
    beam_counts: list[int],
    gold_in_beams: list[bool],
    gold_at_rank0: list[bool],
    parseable_counts: list[int],
    items_any_parseable: list[bool],
) -> dict:
    mean_beams   = statistics.mean(beam_counts)   if beam_counts else 0.0
    median_beams = statistics.median(beam_counts) if beam_counts else 0.0

    gold_hit_count   = sum(gold_in_beams)
    rank0_hit_count  = sum(gold_at_rank0)

    total_beams         = sum(beam_counts)
    total_parseable     = sum(parseable_counts)
    any_parseable_count = sum(items_any_parseable)

    return {
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "dataset":             args.dataset,
        "split":               args.split,
        "mode":                args.mode,
        "kb":                  args.kb,
        "model_id":            model_id,
        "num_beams_requested": args.num_beams,
        "max_new_tokens":      args.max_new_tokens,
        "num_items":           num_items,

        # --- exact match ---
        "gold_in_beams_count": gold_hit_count,
        "gold_in_beams_pct":   round(gold_hit_count / num_items * 100, 2) if num_items else 0.0,
        "gold_at_rank0_count": rank0_hit_count,
        "gold_at_rank0_pct":   round(rank0_hit_count / num_items * 100, 2) if num_items else 0.0,

        # --- beam count distribution ---
        "mean_beams_per_item":   round(mean_beams, 3),
        "median_beams_per_item": median_beams,
        "min_beams":             min(beam_counts) if beam_counts else 0,
        "max_beams":             max(beam_counts) if beam_counts else 0,

        # --- parseability ---
        "total_beams_generated":        total_beams,
        "total_parseable_beams":        total_parseable,
        "parseable_beam_pct":           round(total_parseable / total_beams * 100, 2) if total_beams else 0.0,
        "items_with_any_parseable":     any_parseable_count,
        "items_with_any_parseable_pct": round(any_parseable_count / num_items * 100, 2) if num_items else 0.0,
    }


# ---------------------------------------------------------------------------
# Main

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["infer_backend"] = "huggingface"

    kb_module      = load_kb_module(args.kb)
    common_prefixes = getattr(kb_module, "COMMON_PREFIXES", {})
    if args.mode == "sparql" and not common_prefixes:
        print("Warning: no COMMON_PREFIXES found in KB module — "
              "SPARQL parseability check may report 0%", file=sys.stderr)

    print("Initialising ChatModel...")
    chat_model = ChatModel(cfg)
    engine = chat_model.engine

    data_dir = os.environ.get("DATA_DIR", "data")
    data = load_dataset(args.dataset, args.split, args.mode, data_dir)
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"Capped to {len(data)} examples")

    model_id = Path(cfg.get("adapter_name_or_path") or cfg["model_name_or_path"]).name
    run_name = f"{args.dataset}_{args.split}.{args.mode}"
    out_dir  = os.path.join(data_dir, args.dataset, "predictions", model_id, "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_name}.json")
    print(f"Output: {out_path}")

    # Per-item accumulators for metadata
    beam_counts:         list[int]  = []
    gold_in_beams:       list[bool] = []
    gold_at_rank0:       list[bool] = []
    parseable_counts:    list[int]  = []
    items_any_parseable: list[bool] = []

    results = []
    for item in tqdm(data):
        messages = [{"role": "user", "content": build_question(item["question"])}]
        preds = generate_beams(engine, messages, args.num_beams, args.max_new_tokens)

        gold = item.get("sexpr_with_labels") or item.get("sexpr", "")

        rank        = _gold_rank(preds, gold)
        n_parseable = _count_parseable(preds, args.mode, common_prefixes)

        beam_counts.append(len(preds))
        gold_in_beams.append(rank is not None)
        gold_at_rank0.append(rank == 0)
        parseable_counts.append(n_parseable)
        items_any_parseable.append(n_parseable > 0)

        results.append({**item, "predict": preds})

    # ------------------------------------------------------------------
    num_items = len(results)
    meta = _build_meta(
        args, model_id, num_items,
        beam_counts, gold_in_beams, gold_at_rank0,
        parseable_counts, items_any_parseable,
    )

    output = {"meta": meta, "items": results}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"  Dataset:          {args.dataset} / {args.split} / {args.mode}")
    print(f"  Model:            {model_id}")
    print(f"  Items:            {num_items}")
    print(f"  Gold @ rank 0:    {meta['gold_at_rank0_count']}  ({meta['gold_at_rank0_pct']}%)")
    print(f"  Gold in beams:    {meta['gold_in_beams_count']}  ({meta['gold_in_beams_pct']}%)")
    print(f"  Mean beams:       {meta['mean_beams_per_item']}")
    print(f"  Median beams:     {meta['median_beams_per_item']}")
    print(f"  Parseable beams:  {meta['total_parseable_beams']}/{meta['total_beams_generated']}  ({meta['parseable_beam_pct']}%)")
    print(f"  Items w/ ≥1 parseable: {meta['items_with_any_parseable']}  ({meta['items_with_any_parseable_pct']}%)")
    print(f"{'='*55}")
    print(f"  Saved to: {out_path}\n")


if __name__ == "__main__":
    main()
