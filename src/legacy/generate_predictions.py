"""
generate_predictions.py

Generate beam search predictions from a finetuned model and save.

Usage:
    python generate_predictions.py --config configs/inference/infer.yaml --dataset WDQL --split test --mode simple
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ---------------------------------------------------------------------------
# Argument parsing

def parse_args():
    parser = argparse.ArgumentParser()

    # Option A: llamafactory infer yaml
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a LlamaFactory inference yaml config")

    # Option B: explicit paths (override config)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--adapter_name_or_path", type=str, default=None)
    parser.add_argument("--template", type=str, default=None)

    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name e.g. WDQL")
    parser.add_argument("--split", type=str, default="test",
                        help="Split e.g. train / dev / test")
    parser.add_argument("--mode", type=str, default="simple",
                        choices=["simple", "jena", "sparql"],
                        help="Logical form mode")

    # Generation
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # Misc
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of examples (useful for quick testing)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config resolution

def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_config(args) -> dict:
    cfg = {}

    if args.config is not None:
        cfg = load_yaml_config(args.config)
        print(f"Loaded config from {args.config}")

    # CLI overrides
    if args.model_name_or_path:
        cfg["model_name_or_path"] = args.model_name_or_path
    if args.adapter_name_or_path:
        cfg["adapter_name_or_path"] = args.adapter_name_or_path
    if args.template:
        cfg["template"] = args.template

    assert "model_name_or_path" in cfg, \
        "model_name_or_path must be set via --config or --model_name_or_path"

    return cfg


# ---------------------------------------------------------------------------
# Dataset loading

def load_dataset(dataset: str, split: str, mode: str, data_dir: str) -> tuple[list[dict], str]:
    """
    Load from data/{dataset}/generation/merged/{dataset}_{split}.{mode}.json
    Returns (data, source_path)
    """
    path = os.path.join(
        data_dir, dataset, "generation", "merged",
        f"{dataset}_{split}.{mode}.json"
    )
    print(f"Loading dataset from: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    return data, path


# ---------------------------------------------------------------------------
# Model loading

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def format_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "user", "content": question},
    ]
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return question


def load_model(cfg: dict, dtype: str):
    torch_dtype  = DTYPE_MAP[dtype]
    model_path   = cfg["model_name_or_path"]
    adapter_path = cfg.get("adapter_name_or_path", None)

    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation

def generate_beam(model, tokenizer, prompt: str, num_beams: int, max_new_tokens: int) -> list[str]:
    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            early_stopping=True,
            length_penalty=1.0,
            repetition_penalty=1.0,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]


def get_model_identifier(cfg: dict) -> str:
    """
    Derive a human-readable identifier from model/adapter paths.
    """
    model_name   = Path(cfg["model_name_or_path"]).name
    adapter_path = cfg.get("adapter_name_or_path", None)
    if adapter_path:
        return Path(adapter_path).name
    return model_name


# ---------------------------------------------------------------------------
# Main

def main():
    args = parse_args()
    cfg  = resolve_config(args)

    # Output path: {DATA_DIR}/{dataset}/predictions/{model_id}/raw/{dataset}_{split}.{mode}.json
    model_id         = get_model_identifier(cfg)
    run_name         = f"{args.dataset}_{args.split}.{args.mode}"
    data_dir         = os.environ.get("DATA_DIR", "data")
    output_dir       = os.path.join(data_dir, args.dataset, "predictions", model_id, "raw")
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, f"{run_name}.json")

    print(f"Model identifier : {model_id}")
    print(f"Predictions path : {predictions_path}")

    data, _ = load_dataset(args.dataset, args.split, args.mode, data_dir)
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"Capped to {len(data)} examples")

    model, tokenizer = load_model(cfg, args.dtype)

    _debug_question = (
        f"Generate a Logical Form query that retrieves the information "
        f"corresponding to the given question. \nQuestion: {{ {data[0]['question']} }}"
    )
    _debug_prompt = format_prompt(_debug_question, tokenizer)
    print("\n=== PROMPT REPR (first example) ===")
    print(repr(_debug_prompt))
    print("=== PROMPT RENDERED ===")
    print(_debug_prompt)
    print("=== END PROMPT DEBUG ===\n")

    print(f"Generating predictions (num_beams={args.num_beams})...")
    augmented = []
    for item in tqdm(data):
        question = (
            f"Generate a Logical Form query that retrieves the information "
            f"corresponding to the given question. \nQuestion: {{ {item['question']} }}"
        )
        prompt = format_prompt(question, tokenizer)
        beam   = generate_beam(
            model, tokenizer, prompt,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )
        augmented.append({**item, "predict": beam})

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(augmented)} predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()