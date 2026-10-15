import argparse
import random

from datasets import load_dataset

from alignment.utils import write_jsonl

HUMAN = "\n\nHuman:"
ASSISTANT = "\n\nAssistant:"


def parse_hh_single_turn(text):
    if text.count(HUMAN) != 1 or text.count(ASSISTANT) != 1:
        return None
    human_part, assistant_part = text.split(ASSISTANT)
    if not human_part.startswith(HUMAN):
        return None
    return human_part[len(HUMAN) :].strip(), assistant_part.strip()


def hh_rows(split):
    dataset = load_dataset("Anthropic/hh-rlhf", split=split)
    for example in dataset:
        chosen = parse_hh_single_turn(example["chosen"])
        rejected = parse_hh_single_turn(example["rejected"])
        if chosen is None or rejected is None or chosen[0] != rejected[0]:
            continue
        yield {"instruction": chosen[0], "chosen": chosen[1], "rejected": rejected[1]}


def ultrafeedback_rows(split):
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    for example in dataset:
        chosen, rejected = example["chosen"], example["rejected"]
        if len(chosen) != 2 or len(rejected) != 2:
            continue
        if chosen[0]["role"] != "user" or chosen[1]["role"] != "assistant":
            continue
        if rejected[1]["role"] != "assistant":
            continue
        if example["score_chosen"] <= example["score_rejected"]:
            continue
        yield {
            "instruction": chosen[0]["content"].strip(),
            "chosen": chosen[1]["content"].strip(),
            "rejected": rejected[1]["content"].strip(),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hh", "ultrafeedback"], required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--train_size", type=int, default=15000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--max_instruction_chars", type=int, default=1000)
    parser.add_argument("--max_response_chars", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data/dpo")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "hh":
        rows_iter = hh_rows(args.split or "train")
    else:
        rows_iter = ultrafeedback_rows(args.split or "train_prefs")

    rows = []
    for row in rows_iter:
        if not row["instruction"] or not row["chosen"] or not row["rejected"]:
            continue
        if row["chosen"] == row["rejected"]:
            continue
        if len(row["instruction"]) > args.max_instruction_chars:
            continue
        if len(row["chosen"]) > args.max_response_chars or len(row["rejected"]) > args.max_response_chars:
            continue
        rows.append(row)

    random.Random(args.seed).shuffle(rows)
    needed = args.train_size + args.val_size
    if len(rows) < needed:
        raise ValueError(f"only {len(rows)} usable pairs, need {needed}")
    val_rows = rows[: args.val_size]
    train_rows = rows[args.val_size : needed]
    write_jsonl(f"{args.output_dir}/{args.dataset}_train.jsonl", train_rows)
    write_jsonl(f"{args.output_dir}/{args.dataset}_val.jsonl", val_rows)
    print(f"{args.dataset}: usable {len(rows)} train {len(train_rows)} val {len(val_rows)}")


if __name__ == "__main__":
    main()
