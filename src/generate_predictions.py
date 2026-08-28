import os
import re
import sys
import json
import time
import argparse
import importlib.util
import statistics
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

import torch
import yaml
from llamafactory.chat import ChatModel

from src.utils.kb import load_kb_module


# ---------------------------------------------------------------------------
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="LlamaFactory inference yaml")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--mode", type=str, default="sparql",
                        choices=["jena", "sparql"])
    parser.add_argument("--kb", type=str, default="freebase",
                        help="KB module name under src/kb/")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--diversity_penalty", type=float, default=0.5,
                        help="Diversity penalty for group beam search. "
                             "Higher values = more diverse but potentially less coherent outputs. "
                             "Recommended: 1.0 for Llama, 0.5 for Qwen")
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


def generate_beams(
    engine,
    messages: list[dict],
    num_beams: int,
    max_new_tokens: int,
    diversity_penalty: float
) -> list[str]:

    tok = engine.tokenizer

    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tok(
        text,
        return_tensors="pt"
    ).to(engine.model.device)

    input_len = inputs["input_ids"].shape[1]

    engine.model.generation_config.cache_implementation = None

    with torch.inference_mode():
        outputs = engine.model.generate(
            **inputs,
            custom_generate="transformers-community/group-beam-search",
            num_beams=num_beams,
            num_beam_groups=num_beams,
            diversity_penalty=diversity_penalty,
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
        tok.decode(
            out[input_len:],
            skip_special_tokens=True
        ).strip()
        for out in outputs
    ]

    # Deduplicate, preserving beam order
    seen = set()
    unique = []

    for p in decoded:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique


# ---------------------------------------------------------------------------
# Metadata builder

def _build_meta(
    args,
    model_id: str,
    num_items: int,
    beam_counts: list[int],
    gold_in_beams: list[bool],
    gold_at_rank0: list[bool],
) -> dict:
    mean_beams   = statistics.mean(beam_counts)   if beam_counts else 0.0
    median_beams = statistics.median(beam_counts) if beam_counts else 0.0

    gold_hit_count   = sum(gold_in_beams)
    rank0_hit_count  = sum(gold_at_rank0)

    total_beams = sum(beam_counts)

    return {
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "dataset":             args.dataset,
        "split":               args.split,
        "mode":                args.mode,
        "kb":                  args.kb,
        "model_id":            model_id,
        "num_beams_requested": args.num_beams,
        "max_new_tokens":      args.max_new_tokens,
        "diversity_penalty":   args.diversity_penalty,
        "num_items":           num_items,

        # exact match
        "gold_in_beams_count": gold_hit_count,
        "gold_in_beams_pct":   round(gold_hit_count / num_items * 100, 2) if num_items else 0.0,
        "gold_at_rank0_count": rank0_hit_count,
        "gold_at_rank0_pct":   round(rank0_hit_count / num_items * 100, 2) if num_items else 0.0,

        # beam count distribution 
        "mean_beams_per_item":   round(mean_beams, 3),
        "median_beams_per_item": median_beams,
        "min_beams":             min(beam_counts) if beam_counts else 0,
        "max_beams":             max(beam_counts) if beam_counts else 0,
    }


# ---------------------------------------------------------------------------
# Per-item statistics helpers

def _gold_rank(predictions: list[str], gold: str) -> int | None:
    """Return the rank (0-indexed) of the gold sexpr in predictions, or None."""
    gold_lower = gold.strip().lower()
    for rank, p in enumerate(predictions):
        if p.strip().lower() == gold_lower:
            return rank
    return None


# ---------------------------------------------------------------------------
# Checkpoint (incremental JSONL) helpers

def load_checkpoint(jsonl_path: str) -> dict[int, dict]:
    done = {}
    if not os.path.isfile(jsonl_path):
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[rec["idx"]] = rec["item"]
    return done


def append_checkpoint(jsonl_path: str, idx: int, item: dict) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"idx": idx, "item": item}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Main

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["infer_backend"] = "huggingface"

    # Load KB module
    kb_instance = load_kb_module(args.kb)

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
    out_path   = os.path.join(out_dir, f"{run_name}.json")
    ckpt_path  = os.path.join(out_dir, f"{run_name}.ckpt.jsonl")
    print(f"Output:     {out_path}")
    print(f"Checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Resume from checkpoint, if present
    done = load_checkpoint(ckpt_path)
    if done:
        print(f"Resuming: {len(done)}/{len(data)} items already completed in checkpoint.")

    # Per-item accumulators for metadata
    results: list[dict | None] = [None] * len(data)

    for idx, item in enumerate(tqdm(data)):
        if idx in done:
            results[idx] = done[idx]
            continue

        messages = [{"role": "user", "content": build_question(item["question"])}]

        preds = generate_beams(
            engine, messages, args.num_beams, args.max_new_tokens,
            args.diversity_penalty
        )

        record = {**item, "predict": preds}
        results[idx] = record
        append_checkpoint(ckpt_path, idx, record)

    # ------------------------------------------------------------------
    # Recompute stats over the full (resumed + fresh) result set
    beam_counts:   list[int]  = []
    gold_in_beams: list[bool] = []
    gold_at_rank0: list[bool] = []

    for record in results:
        preds = record["predict"]
        gold = record.get("sexpr_with_labels") or record.get("sexpr", "")

        rank = _gold_rank(preds, gold)

        beam_counts.append(len(preds))
        gold_in_beams.append(rank is not None)
        gold_at_rank0.append(rank == 0)

    num_items = len(results)
    meta = _build_meta(
        args, model_id, num_items,
        beam_counts, gold_in_beams, gold_at_rank0,
    )

    output = {"meta": meta, "items": results}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"  Dataset:              {args.dataset} / {args.split} / {args.mode}")
    print(f"  Model:                {model_id}")
    print(f"  Items:                {num_items}")
    print(f"  Diversity penalty:    {args.diversity_penalty}")
    print(f"  Gold @ rank 0:        {meta['gold_at_rank0_count']}  ({meta['gold_at_rank0_pct']}%)")
    print(f"  Gold in beams:        {meta['gold_in_beams_count']}  ({meta['gold_in_beams_pct']}%)")
    print(f"  Mean beams:           {meta['mean_beams_per_item']}")
    print(f"  Median beams:         {meta['median_beams_per_item']}")
    print(f"{'='*55}")
    print(f"  Saved to: {out_path}\n")


if __name__ == "__main__":
    main()