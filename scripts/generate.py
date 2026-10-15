import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.prompts import format_prompt
from alignment.utils import append_jsonl, read_jsonl, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_prompt_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--chat_template", action="store_true")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    with open(args.prompts_path, "r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    partial_path = args.output_path + ".partial.jsonl"
    done = {row["id"]: row for row in read_jsonl(partial_path)} if os.path.exists(partial_path) else {}
    pending = [item for item in prompts if item["id"] not in done]
    print(f"{len(done)} already generated, {len(pending)} to go", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    model = model.to(device)
    model.eval()

    def render(instruction):
        if args.chat_template:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
            )
        return format_prompt(instruction)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.repetition_penalty != 1.0:
        generation_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size > 0:
        generation_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start : start + args.batch_size]
        texts = [render(item["instruction"]) for item in chunk]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_tokens,
            add_special_tokens=not args.chat_template,
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(**encoded, **generation_kwargs)
        new_tokens = output_ids[:, encoded["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for item, text in zip(chunk, decoded):
            row = {
                "id": item["id"],
                "instruction": item["instruction"],
                "output": text.strip(),
                "generator": args.generator,
                "dataset": args.dataset or item.get("dataset", "unknown"),
            }
            append_jsonl(partial_path, row)
            done[item["id"]] = row
        print(f"{len(done)}/{len(prompts)}", flush=True)

    ordered = [done[item["id"]] for item in prompts]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(ordered, handle, indent=2, ensure_ascii=False)
    print(f"wrote {len(ordered)} outputs to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
