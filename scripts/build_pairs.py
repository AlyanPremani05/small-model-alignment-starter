import argparse
import json
import random

from alignment.utils import write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_path", required=True)
    parser.add_argument("--baseline_path", required=True)
    parser.add_argument("--subject_name", default="dpo")
    parser.add_argument("--baseline_name", default="sft")
    parser.add_argument("--prompts_path", default=None)
    parser.add_argument("--kind", choices=["general", "safety", "persona"], default="general")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return {row["id"]: row for row in rows}


def main():
    args = parse_args()
    subject = load(args.subject_path)
    baseline = load(args.baseline_path)
    if set(subject) != set(baseline):
        raise ValueError("subject and baseline outputs must cover exactly the same prompt ids")

    categories = {}
    if args.prompts_path:
        with open(args.prompts_path, "r", encoding="utf-8") as handle:
            categories = {item["id"]: item.get("category") for item in json.load(handle)}

    rng = random.Random(args.seed)
    rows = []
    for prompt_id in subject:
        subject_first = rng.random() < 0.5
        first, second = (subject, baseline) if subject_first else (baseline, subject)
        first_name, second_name = (
            (args.subject_name, args.baseline_name) if subject_first else (args.baseline_name, args.subject_name)
        )
        rows.append(
            {
                "pair_id": prompt_id,
                "instruction": subject[prompt_id]["instruction"],
                "response_a": first[prompt_id]["output"],
                "response_b": second[prompt_id]["output"],
                "kind": args.kind,
                "category": categories.get(prompt_id),
                "hidden": {"a_model": first_name, "b_model": second_name},
            }
        )
    write_jsonl(args.output_path, rows)
    a_is_subject = sum(1 for row in rows if row["hidden"]["a_model"] == args.subject_name)
    print(f"wrote {len(rows)} pairs to {args.output_path} ({a_is_subject} with {args.subject_name} as A)")


if __name__ == "__main__":
    main()
