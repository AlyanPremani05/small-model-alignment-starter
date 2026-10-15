import argparse
import json
import os
import random

from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default="eval_sets/alpaca_eval_300_seed42.json")
    return parser.parse_args()


def main():
    args = parse_args()
    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    indices = random.Random(args.seed).sample(range(len(data)), args.num_prompts)
    subset = [
        {
            "id": f"alpaca_eval-{index}",
            "instruction": data[index]["instruction"],
            "dataset": "alpaca_eval",
            "category": data[index]["dataset"],
        }
        for index in indices
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(subset, handle, indent=2, ensure_ascii=False)
    print(f"wrote {len(subset)} prompts to {args.output_path}")


if __name__ == "__main__":
    main()
