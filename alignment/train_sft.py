import argparse
import contextlib
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.core import build_lm_batch, sequence_logps, sft_loss_from_batch
from alignment.prompts import format_prompt
from alignment.utils import append_jsonl, cosine_with_warmup, read_jsonl, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_pairs(path, limit):
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    return [(format_prompt(row["instruction"]), row["response"]) for row in rows]


def make_collate(tokenizer, max_length):
    def collate(items):
        prompts = [item[0] for item in items]
        responses = [item[1] for item in items]
        return build_lm_batch(tokenizer, prompts, responses, max_length)

    return collate


@torch.no_grad()
def evaluate(model, loader, autocast_ctx):
    model.eval()
    total_logp = 0.0
    total_tokens = 0
    for batch in loader:
        with autocast_ctx():
            logps, counts = sequence_logps(model, batch)
        total_logp += logps.sum().item()
        total_tokens += counts.sum().item()
    model.train()
    return -total_logp / max(1, total_tokens)


def save_checkpoint(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    use_amp = device.type == "cuda"

    def autocast_ctx():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    train_pairs = load_pairs(args.train_path, args.max_train_examples)
    val_pairs = load_pairs(args.val_path, args.max_val_examples)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_pairs,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer, args.max_length),
        generator=generator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_pairs,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, args.max_length),
    )

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(args.warmup_ratio * total_steps)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 or "norm" in name or "bias" in name else decay).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    scheduler = cosine_with_warmup(optimizer, total_steps, warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    best_val = math.inf
    step = 0
    micro = 0
    running_loss = 0.0
    running_count = 0
    start = time.time()

    def run_eval():
        nonlocal best_val
        val_loss = evaluate(model, val_loader, autocast_ctx)
        append_jsonl(metrics_path, {"type": "eval", "step": step, "val_loss": val_loss})
        print(f"step {step} val_loss {val_loss:.4f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "best"))
            append_jsonl(metrics_path, {"type": "best", "step": step, "val_loss": val_loss})

    done = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for batch in train_loader:
            with autocast_ctx():
                loss = sft_loss_from_batch(model, batch)
            scaler.scale(loss / args.grad_accum).backward()
            running_loss += loss.item()
            running_count += 1
            micro += 1
            if micro % args.grad_accum != 0:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                row = {
                    "type": "train",
                    "step": step,
                    "epoch": epoch,
                    "train_loss": running_loss / running_count,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "elapsed": time.time() - start,
                }
                append_jsonl(metrics_path, row)
                print(f"step {step}/{total_steps} loss {row['train_loss']:.4f} lr {row['lr']:.2e}", flush=True)
                running_loss = 0.0
                running_count = 0

            if step % args.eval_every == 0:
                run_eval()

            if step >= total_steps:
                done = True
                break
        if done:
            break

    if step % args.eval_every != 0:
        run_eval()
    save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "final"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
