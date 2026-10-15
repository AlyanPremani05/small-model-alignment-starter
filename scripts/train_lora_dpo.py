import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import dataclasses
import json
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--data_path", default="data/persona_lora/ramsay_style_preferences.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_size", type=int, default=20)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def supported(config_cls, values):
    names = {field.name for field in dataclasses.fields(config_cls)}
    return {key: value for key, value in values.items() if key in names}


def to_record(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()

    with open(args.data_path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    random.Random(args.seed).shuffle(rows)
    val_rows, train_rows = rows[: args.val_size], rows[args.val_size :]
    train_dataset = Dataset.from_list([to_record(row) for row in train_rows])
    eval_dataset = Dataset.from_list([to_record(row) for row in val_rows])
    print(f"train {len(train_rows)} val {len(val_rows)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=[name.strip() for name in args.target_modules.split(",")],
        bias="none",
        task_type="CAUSAL_LM",
    )

    config_values = {
        "output_dir": args.output_dir,
        "beta": args.beta,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "logging_steps": 5,
        "eval_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "save_strategy": "no",
        "report_to": "none",
        "fp16": use_cuda,
        "bf16": False,
        "gradient_checkpointing": False,
        "seed": args.seed,
        "remove_unused_columns": False,
    }
    if "warmup_ratio" in {field.name for field in dataclasses.fields(DPOConfig)}:
        config_values["warmup_ratio"] = args.warmup_ratio
    else:
        config_values["warmup_steps"] = args.warmup_ratio
    config = DPOConfig(**supported(DPOConfig, config_values))

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable parameters {trainable} of {total} ({100 * trainable / total:.2f}%)", flush=True)

    trainer.train()
    final_eval = trainer.evaluate()
    print(final_eval, flush=True)

    adapter_dir = os.path.join(args.output_dir, "adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"log_history": trainer.state.log_history, "final_eval": final_eval, "args": vars(args)},
            handle,
            indent=2,
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
