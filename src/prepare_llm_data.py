import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path

from src.utils.run_config import apply_run_config_defaults, require

# ---------------------------------------------------------------------------
# Arg handling

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset name")
    parser.add_argument('--split', default='train', type=str, help="Split to process")
    parser.add_argument('--run_config', type=str, help="Path to configs/run/<name>.yaml")

    apply_run_config_defaults(parser, section="prepare")

    args = parser.parse_args()
    require(args, "dataset")
    return args


# ---------------------------------------------------------------------------
# File handling

def load_data(split, args):
    """
    Loads the target dataset's label-enriched train split files for every training target.
    """
    data_dir = os.getenv("DATA_DIR", "data")
    base = Path(f"{data_dir}/{args.dataset}/generation/merged")
    # Search for any training target files
    pattern = f"{args.dataset}_{split}.*.json"
    files = list(base.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    data_by_mode = {}

    for f in files:
        name = f.stem  # dataset_split.mode
        parts = name.split(".")
        mode = parts[-1]

        print("[INFO] Loading:", f)
        with open(f, encoding="utf-8") as fh:
            data_by_mode[mode] = json.load(fh)

    return data_by_mode

# ---------------------------------------------------------------------------
# Process

def prepare_dataloader(args, split):
    """
    For a given label-enriched split file, generate a Llamafactory training dataset.
    """
    data_by_mode = load_data(split, args)

    for mode, data in data_by_mode.items():

        print(f"\n[INFO] === Mode: {mode} ===")
        print(f'[INFO] Input {split} split len: {len(data)}')

        # Filter empty outputs
        before = len(examples)
        examples = [x for x in data if x.get('sexpr_with_labels', '').strip()]
        print(f'[WARN] Dropped {before - len(examples)} entries with empty sexpr_with_labels')
        print(f'[INFO] Real {split} dataset len: {len(examples)}')

        # Construct dataset format expected by Llamafactory
        instruction = 'Generate a Logical Form query that retrieves the information corresponding to the given question. \n'
        json_data = []
        for item in tqdm(examples):
            json_data.append({
                "instruction": instruction,
                "input": 'Question: { ' + item['question'] + ' }',
                "output": item['sexpr_with_labels'],
                "history": [],
            })

        # Save dataset to disk
        llm_dir = os.getenv("LLM_DIR", "LLMs")
        output_dir = f'{llm_dir}/data/{args.dataset}_{split}.{mode}/examples.json'
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(output_dir, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
        print(f'Written {len(json_data)} examples to {output_dir}')

        # Register the dataset so Llamafactory can use it
        register_dataset(args.dataset, f"train.{mode}")


LLM_DIR = os.getenv("LLM_DIR", "LLMs")
DATASET_INFO_PATH = f'{LLM_DIR}/data/dataset_info.json'

def register_dataset(dataset: str, split: str) -> None:
    """
    Appends an entry to Llamafactory's dataset_info.json.
    This specifies the format and path, so it can be used in Llamafactory training configs
    by simply entering its name only.
    """
    # Construct entry
    key = f'{dataset}_{split}'
    entry = {
        "file_name": f'{dataset}_{split}/examples.json',
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": "history"
        }
    }

    if os.path.exists(DATASET_INFO_PATH):
        with open(DATASET_INFO_PATH, encoding='utf-8') as f:
            info = json.load(f)
    else:
        info = {}

    # Already registered
    if info.get(key) == entry:
        return

    # Append entry
    info[key] = entry
    with open(DATASET_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f'Registered "{key}" in {DATASET_INFO_PATH}')


if __name__ == '__main__':
    args = _parse_args()
    prepare_dataloader(args, "train")