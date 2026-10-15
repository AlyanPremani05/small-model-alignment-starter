import argparse
import random

from datasets import load_dataset

from alignment.utils import write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--train_size", type=int, default=30000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--max_instruction_chars", type=int, default=1500)
    parser.add_argument("--max_response_chars", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data/sft")
    return parser.parse_args()


def first_turn(messages):
    if len(messages) < 2:
        return None
    first, second = messages[0], messages[1]
    if first["role"] != "user" or second["role"] != "assistant":
        return None
    return first["content"].strip(), second["content"].strip()


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset, split=args.split)
    rows = []
    for example in dataset:
        turn = first_turn(example["messages"])
        if turn is None:
            continue
        instruction, response = turn
        if not instruction or not response:
            continue
        if len(instruction) > args.max_instruction_chars or len(response) > args.max_response_chars:
            continue
        rows.append({"instruction": instruction, "response": response})

    random.Random(args.seed).shuffle(rows)
    needed = args.train_size + args.val_size
    if len(rows) < needed:
        raise ValueError(f"only {len(rows)} usable examples, need {needed}")
    val_rows = rows[: args.val_size]
    train_rows = rows[args.val_size : needed]
    write_jsonl(f"{args.output_dir}/sft_train.jsonl", train_rows)
    write_jsonl(f"{args.output_dir}/sft_val.jsonl", val_rows)
    print(f"train {len(train_rows)} val {len(val_rows)}")


if __name__ == "__main__":
    main()
