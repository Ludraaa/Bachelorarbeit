import argparse
import json
import sys


def load_items(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    raise ValueError(f"Unrecognized file structure in {path!r}")


def get_id(item):
    return item.get("qid") or item.get("ID")


def get_question(item):
    q = item.get("question")
    return q.strip() if isinstance(q, str) else q


def get_f1(item):
    return item.get("f1", item.get("assignment_f1"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument(
        "--a_better_only",
        action="store_true"
    )
    parser.add_argument(
        "--by",
        choices=["id", "question"],
        default="id",
        help="Key to match items on across the two files. Use 'question' "
             "when IDs differ between file_a and file_b but the question "
             "text is shared (default: id).",
    )
    args = parser.parse_args()

    get_key = get_question if args.by == "question" else get_id

    f1_a = {get_key(i): get_f1(i) for i in load_items(args.file_a) if get_key(i) is not None}
    f1_b = {get_key(i): get_f1(i) for i in load_items(args.file_b) if get_key(i) is not None}

    all_ids = set(f1_a) | set(f1_b)

    diffs = []
    for id_ in all_ids:
        a = float(f1_a.get(id_, 0.0))
        b = float(f1_b.get(id_, 0.0))
        d = abs(a - b)
        if d <= args.epsilon:
            continue
        if args.a_better_only and a <= b:
            continue
        diffs.append((id_, a, b, d))

    diffs.sort(key=lambda r: r[3], reverse=True)

    for id_, a, b, d in diffs:
        print(f"{id_:60s} | A={a:.6f}  B={b:.6f}  diff={d:.4f}")

    print(f"\n{len(diffs)}/{len(all_ids)} differ", file=sys.stderr)


if __name__ == "__main__":
    main()