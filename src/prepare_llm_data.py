import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path

from src.utils.run_config import apply_run_config_defaults, require

# ---------------------------------------------------------------------------
# args

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str,
                        help="Dataset name, e.g. WebQSP — resolves to data/{dataset}/generation/merged/")
    parser.add_argument('--split', default='train', type=str,
                        help="Split to process: train | dev | test (default: train)")
    parser.add_argument('--run_config', default=None, type=str,
                        help="Path to configs/run/<name>.yaml; values become defaults, "
                             "explicit flags still override.")

    apply_run_config_defaults(parser, section="prepare")

    args = parser.parse_args()
    require(args, "dataset")
    return args


# ---------------------------------------------------------------------------
# file stuff

def load_data(split, args):
    data_dir = os.getenv("DATA_DIR", "data")
    base = Path(f"{data_dir}/{args.dataset}/generation/merged")
    pattern = f"{args.dataset}_{split}.*.json"

    files = list(base.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    data_by_mode = {}

    for f in files:
        name = f.stem  # dataset_split.method
        parts = name.split(".")
        mode = parts[-1]

        print("Loading:", f)
        with open(f, encoding="utf-8") as fh:
            data_by_mode[mode] = json.load(fh)

    return data_by_mode

# ---------------------------------------------------------------------------
# processing

def prepare_dataloader(args, split):
    data_by_mode = load_data(split, args)

    for mode, data in data_by_mode.items():

        print(f"\n=== Mode: {mode} ===")
        print(f'Origin {split} dataset len: {len(data)}')
        assert type(data) == list

        if 'sexpr_with_labels' not in data[0]:
            raise KeyError(
                "'sexpr_with_labels' field missing — has insert_labels.py been run on this dataset?"
            )

        if 'train' in split or 'dev' in split:
            examples = [x for x in data if x['sexpr'].lower() != 'null']
        else:
            examples = list(data)

        # Filter empty outputs
        before = len(examples)
        examples = [x for x in examples if x.get('sexpr_with_labels', '').strip()]
        print(f'Dropped {before - len(examples)} entries with empty sexpr_with_labels')
        
        print(f'Real {split} dataset len: {len(examples)}')

        instruction = 'Generate a Logical Form query that retrieves the information corresponding to the given question. \n'
        json_data = []
        for item in tqdm(examples):
            json_data.append({
                "instruction": instruction,
                "input": 'Question: { ' + item['question'] + ' }',
                "output": item['sexpr_with_labels'],
                "history": [],
            })

        llm_dir = os.getenv("LLM_DIR", "LLMs")
        output_dir = f'{llm_dir}/data/{args.dataset}_{split}.{mode}/examples.json'
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(output_dir, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
        print(f'Written {len(json_data)} examples to {output_dir}')

        register_dataset(args.dataset, f"train.{mode}")

LLM_DIR = os.getenv("LLM_DIR", "LLMs")
DATASET_INFO_PATH = f'{LLM_DIR}/data/dataset_info.json'

def register_dataset(dataset: str, split: str) -> None:
    key = f'{dataset}_{split}'
    entry = {
        "file_name": f'{dataset}_{split}/examples.json',  # {LLM_DIR}/data/...
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

    if info.get(key) == entry:
        return  # already registered

    info[key] = entry

    with open(DATASET_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f'Registered "{key}" in {DATASET_INFO_PATH}')

if __name__ == '__main__':
    args = _parse_args()
    print(args)
    prepare_dataloader(args, "train")
    print('Finished')