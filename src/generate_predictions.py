import os
import json
import argparse
import statistics
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm
import torch
import yaml
from llamafactory.chat import ChatModel

from src.utils.run_config import apply_run_config_defaults, require, validate_choice


# ---------------------------------------------------------------------------
# Arg handling

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="LlamaFactory inference yaml")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--mode", type=str, default="sparql")
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--diversity_penalty", type=float, default=0.5,
                        help="Diversity penalty for group beam search. "
                             "Higher values = more diverse but potentially less coherent outputs. "
                             "Recommended: ~1.0 for Llama, ~0.5 for Qwen")
    parser.add_argument("--run_config", type=str)
    parser.add_argument("--oracle", action="store_true",
                        help="Skip inference and directly output the ground truth (sexpr_with_labels) as the single prediction.")

    apply_run_config_defaults(parser, section="generate", config_ref_key="infer_config")

    args = parser.parse_args()
    require(args, "config", "dataset",)
    validate_choice(args, "mode", ["jena", "sparql"])
    return args


# ---------------------------------------------------------------------------
# Dataset

def load_dataset(dataset, split, mode, data_dir):
    """
    Loads the specified label-enriched split file to use for predictions.
    """
    path = os.path.join(data_dir, dataset, "generation", "merged", f"{dataset}_{split}.{mode}.json")
    print(f"[INFO] Loading dataset: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO]   {len(data)} examples")
    return data


# ---------------------------------------------------------------------------
# Generation

INSTRUCTION = (
    "Generate a Logical Form query that retrieves the information corresponding to the given question."
)


def build_question(raw_question: str) -> str:
    """
    Formats the question exactly like in the training data.
    """
    return f"{INSTRUCTION}\n\nQuestion: {{ {raw_question} }}"


def generate_beams(
    engine,
    messages: list[dict],
    num_beams: int,
    max_new_tokens: int,
    diversity_penalty: float
) -> list[str]:
    """
    Generate requested number of prediction beams using group beam search.
    Duplicate beams are removed from the prediction list at the end.
    """

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
# Run identity

def _run_manifest_dict(args) -> dict:
    """
    Defines dict of parameters that determine the content of a generation run.
    This is used for continuation logic if an existing run was interrupted.
    """
    return {
        "dataset": args.dataset,
        "split": args.split,
        "mode": args.mode,
        "kb": args.kb,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "diversity_penalty": args.diversity_penalty,
        "oracle": args.oracle,
    }


def _check_or_write_manifest(run_dir: str, manifest: dict) -> None:
    """
    Looks for an existing run manifest in the output folder.
    If no manifets exists, write the current run's manifest to disk.
    If one does exist, do nothing unless it is different from the 
    current run's manifest.
    """
    path = os.path.join(run_dir, "run_manifest.json")
    if os.path.exists(path):
        existing = json.loads(Path(path).read_text(encoding="utf-8"))
        
        # Handle resume for older manifests missing the oracle key
        if "oracle" not in existing:
            existing["oracle"] = False
        
        # Relevant config parameter is different
        if existing != manifest:
            raise ValueError(
                f"Run folder already exists with different parameters: {run_dir}\n"
                f"Existing:  {json.dumps(existing, sort_keys=True)}\n"
                f"Requested: {json.dumps(manifest, sort_keys=True)}\n"
                f"Use a different run_config (or delete the folder to start over)."
            )
    else:
        # Write current run manifest
        os.makedirs(run_dir, exist_ok=True)
        Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _reset_if_already_finished(out_path: str, ckpt_path: str) -> None:
    """
    If an existing run is found and continue logic would be triggered,
    check whether that run is already finished or if it actually needs
    continuation. If it is done, assume the user wants to run the script
    again (instead of having to delete the produced file manually).
    """
    if os.path.exists(out_path):
        print(f"[INFO] Final output already exists ({out_path}). Starting fresh.")
        os.remove(out_path)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


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
        "oracle":              args.oracle,
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
# Statistics helpers

def _gold_rank(predictions: list[str], gold: str) -> int | None:
    """
    Return the rank of the gold sexpr in the predictions list, or None.
    """
    gold_lower = gold.strip().lower()
    for rank, p in enumerate(predictions):
        if p.strip().lower() == gold_lower:
            return rank
    return None


# ---------------------------------------------------------------------------
# Incremental write helpers

def load_checkpoint(jsonl_path: str) -> dict[int, dict]:
    """
    Loads the specified jsonl checkpoint of the current run. The script will
    continue processing from where the checkpoint left off.
    """
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
    """
    Appends a new item to the incremental jsonl file.
    """
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"idx": idx, "item": item}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Build infer config

def build_chat_config(infer_config_path: str, training_config_path: str | None) -> dict:
    """
    Layers inference-specific settings (infer_dtype, trust_remote_code, ...) on
    top of the training config's model identity, so model_name_or_path /
    adapter path / template do not need to be declared twice, once in training
    config and once in inference config.
    """
    with open(infer_config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if training_config_path:
        with open(training_config_path, encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f)
        cfg.setdefault("model_name_or_path", train_cfg["model_name_or_path"])
        cfg.setdefault("adapter_name_or_path", train_cfg.get("output_dir"))
        cfg.setdefault("finetuning_type", train_cfg.get("finetuning_type"))
        cfg.setdefault("template", train_cfg.get("template"))

    cfg["infer_backend"] = "huggingface"
    return cfg


# ---------------------------------------------------------------------------
# Main

def main():
    args = parse_args()

    run_cfg = {}
    if args.run_config:
        with open(args.run_config, encoding="utf-8") as f:
            run_cfg = yaml.safe_load(f) or {}

    cfg = build_chat_config(args.config, run_cfg.get("training_config"))

    # Determine the model_id
    model_id = Path(cfg.get("adapter_name_or_path") or cfg["model_name_or_path"]).name

    if args.oracle:
        print("[INFO] Oracle mode enabled. Skipping model initialization.")
        engine = None
    else:
        print("[INFO] Initializing ChatModel...")
        chat_model = ChatModel(cfg)
        engine = chat_model.engine

    data_dir = os.environ.get("DATA_DIR", "data")
    data = load_dataset(args.dataset, args.split, args.mode, data_dir)
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"[WARN] Capped to {len(data)} examples")

    # Output path logic
    # configs/runs/Wikidata/Qald7/sparql.yaml -> "sparql"
    run_stem = Path(args.run_config).stem
    run_name = f"{args.dataset}_{args.split}.{args.mode}"
    out_dir  = os.path.join(data_dir, args.dataset, "predictions", model_id, run_stem, "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path   = os.path.join(out_dir, f"{run_name}.json")
    ckpt_path  = os.path.join(out_dir, f"{run_name}.ckpt.jsonl")

    _reset_if_already_finished(out_path, ckpt_path)

    _check_or_write_manifest(out_dir, _run_manifest_dict(args))

    print(f"[INFO] Output:     {out_path}")
    print(f"[INFO] Checkpoint: {ckpt_path}")

    # Resume from checkpoint, if present
    done = load_checkpoint(ckpt_path)
    if done:
        print(f"[INFO] Resuming: {len(done)}/{len(data)} items already completed in checkpoint.")

    # Accumulator
    results: list[dict | None] = [None] * len(data)

    for idx, item in enumerate(tqdm(data)):
        # Load processed items
        if idx in done:
            results[idx] = done[idx]
            continue

        # Oracle flag handling, no actual inference required
        if args.oracle:
            preds = [item.get("sexpr_with_labels") or item.get("sexpr", "")]
        # Beam generation
        else:
            messages = [{"role": "user", "content": build_question(item["question"])}]
            preds = generate_beams(
                engine, messages, args.num_beams, args.max_new_tokens,
                args.diversity_penalty
            )

        # Incremental write
        record = {**item, "predict": preds}
        results[idx] = record
        append_checkpoint(ckpt_path, idx, record)

    # ------------------------------------------------------------------
    # Compute stats over the full result set
    beam_counts: list[int]  = []
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
    meta = _build_meta(args, model_id, num_items, beam_counts, gold_in_beams, gold_at_rank0)

    output = {"meta": meta, "items": results}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print run statistics
    print(f"\n{'-'*55}")
    print(f"  Dataset:              {args.dataset} / {args.split} / {args.mode}")
    print(f"  Model:                {model_id}")
    if args.oracle:
        print(f"  Mode:                 Oracle (Ground Truth)")
    print(f"  Items:                {num_items}")
    print(f"  Diversity penalty:    {args.diversity_penalty}")
    print(f"  Gold @ rank 0:        {meta['gold_at_rank0_count']}  ({meta['gold_at_rank0_pct']}%)")
    print(f"  Gold in beams:        {meta['gold_in_beams_count']}  ({meta['gold_in_beams_pct']}%)")
    print(f"  Mean beams:           {meta['mean_beams_per_item']}")
    print(f"  Median beams:         {meta['median_beams_per_item']}")

    print(f"\n[INFO] Saved to: {out_path}\n")


if __name__ == "__main__":
    main()