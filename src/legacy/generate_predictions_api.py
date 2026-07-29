"""
generate_predictions_api.py
"""

import os
import json
import argparse
import requests
from tqdm import tqdm
from pathlib import Path


# ---------------------------------------------------------------------------
# Args

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--mode", type=str, default="simple")

    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset

def load_dataset(dataset, split, mode, data_dir):
    path = os.path.join(
        data_dir, dataset, "generation", "merged",
        f"{dataset}_{split}.{mode}.json"
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, path


# ---------------------------------------------------------------------------
# API call

def call_llamafactory_api(api_base, api_key, prompt, num_beams=5):
    url = f"{api_base}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "n": num_beams,
        "max_tokens": 512,
    }

    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    data = r.json()

    return [c["message"]["content"].strip() for c in data["choices"]]


# ---------------------------------------------------------------------------
# Prompt

def build_prompt(question: str):
    return (
        "Generate a Logical Form query that retrieves the information corresponding to the given question.\n\n"
        f"Question: {{ {question} }}"
    )


# ---------------------------------------------------------------------------
# Main

def main():
    args = parse_args()

    data_dir = os.environ.get("DATA_DIR", "data")

    data, _ = load_dataset(args.dataset, args.split, args.mode, data_dir)

    if args.max_samples:
        data = data[:args.max_samples]

    run_name = f"{args.dataset}_{args.split}.{args.mode}"
    output_dir = os.path.join(data_dir, args.dataset, "predictions_api")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{run_name}.json")

    results = []

    print(f"Running API inference on {len(data)} samples...")

    for item in tqdm(data):
        question = item["question"]
        prompt = build_prompt(question)

        preds = call_llamafactory_api(
            args.api_base,
            args.api_key,
            prompt,
            num_beams=args.num_beams
        )

        results.append({
            **item,
            "predict": preds
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()