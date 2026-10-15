import argparse
import contextlib
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.core import (
    build_dpo_batch,
    dpo_loss_from_batch,
    dpo_loss_from_logps,
    dpo_reward_stats,
    sequence_logps,
)
from alignment.prompts import format_prompt
from alignment.utils import append_jsonl, read_jsonl, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--reference_path", default=None)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy_device", default=None)
    parser.add_argument("--reference_device", default=None)
    return parser.parse_args()


def default_devices(args):
    if torch.cuda.is_available():
        policy = args.policy_device or "cuda:0"
        reference = args.reference_device or ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    else:
        policy = args.policy_device or "cpu"
        reference = args.reference_device or "cpu"
    return torch.device(policy), torch.device(reference)


def load_pairs(path, limit, tokenizer, max_prompt_length):
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    kept = []
    for row in rows:
        prompt = format_prompt(row["instruction"])
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > max_prompt_length:
            continue
        kept.append((prompt, row["chosen"], row["rejected"]))
    print(f"{path}: kept {len(kept)} of {len(rows)} pairs after prompt-length filter", flush=True)
    return kept


def make_collate(tokenizer, max_length):
    def collate(items):
        return build_dpo_batch(
            tokenizer,
            [item[0] for item in items],
            [item[1] for item in items],
            [item[2] for item in items],
            max_length,
        )

    return collate


@torch.no_grad()
def reference_logps_for(model, loader):
    chosen, rejected = [], []
    for batch in loader:
        c, _ = sequence_logps(model, batch["chosen"])
        r, _ = sequence_logps(model, batch["rejected"])
        chosen.append(c.float().cpu())
        rejected.append(r.float().cpu())
    return torch.cat(chosen), torch.cat(rejected)


@torch.no_grad()
def evaluate(policy, loader, ref_chosen, ref_rejected, beta, autocast_ctx):
    policy.eval()
    chosen, rejected = [], []
    for batch in loader:
        with autocast_ctx():
            c, _ = sequence_logps(policy, batch["chosen"])
            r, _ = sequence_logps(policy, batch["rejected"])
        chosen.append(c.float().cpu())
        rejected.append(r.float().cpu())
    policy.train()
    pol_c, pol_r = torch.cat(chosen), torch.cat(rejected)
    loss = dpo_loss_from_logps(pol_c, pol_r, ref_chosen, ref_rejected, beta).item()
    stats = dpo_reward_stats(pol_c, pol_r, ref_chosen, ref_rejected, beta)
    stats["val_loss"] = loss
    return stats


def save_checkpoint(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def main():
    args = parse_args()
    set_seed(args.seed)
    policy_device, reference_device = default_devices(args)
    use_amp = policy_device.type == "cuda"

    def policy_autocast():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    tokenizer = AutoTokenizer.from_pretrained(args.policy_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(args.policy_path, torch_dtype=torch.float32)
    policy.config.use_cache = False
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    policy.to(policy_device)
    policy.train()

    reference_dtype = torch.float16 if reference_device.type == "cuda" else torch.float32
    reference = AutoModelForCausalLM.from_pretrained(
        args.reference_path or args.policy_path, torch_dtype=reference_dtype
    )
    reference.to(reference_device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    train_pairs = load_pairs(args.train_path, args.max_train_examples, tokenizer, args.max_prompt_length)
    val_pairs = load_pairs(args.val_path, args.max_val_examples, tokenizer, args.max_prompt_length)
    collate = make_collate(tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_pairs, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=generator, drop_last=True
    )
    val_loader = DataLoader(val_pairs, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    ref_val_chosen, ref_val_rejected = reference_logps_for(reference, val_loader)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    best_key = (-1.0, float("-inf"))
    step = 0
    micro = 0
    running = {"loss": 0.0, "reward_accuracy": 0.0, "reward_margin": 0.0}
    running_count = 0
    start = time.time()

    def run_eval(save_best):
        nonlocal best_key
        stats = evaluate(policy, val_loader, ref_val_chosen, ref_val_rejected, args.beta, policy_autocast)
        append_jsonl(metrics_path, {"type": "eval", "step": step, **stats})
        print(
            f"step {step} val_loss {stats['val_loss']:.4f} acc {stats['reward_accuracy']:.3f} "
            f"margin {stats['reward_margin']:.4f}",
            flush=True,
        )
        key = (stats["reward_accuracy"], stats["reward_margin"])
        if save_best and key > best_key:
            best_key = key
            save_checkpoint(policy, tokenizer, os.path.join(args.output_dir, "best"))
            append_jsonl(metrics_path, {"type": "best", "step": step, "reward_accuracy": key[0], "reward_margin": key[1]})

    run_eval(save_best=False)

    done = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for batch in train_loader:
            with policy_autocast():
                loss, stats = dpo_loss_from_batch(policy, reference, batch, args.beta)
            scaler.scale(loss / args.grad_accum).backward()
            running["loss"] += loss.item()
            running["reward_accuracy"] += stats["reward_accuracy"]
            running["reward_margin"] += stats["reward_margin"]
            running_count += 1
            micro += 1
            if micro % args.grad_accum != 0:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                row = {
                    "type": "train",
                    "step": step,
                    "epoch": epoch,
                    "train_loss": running["loss"] / running_count,
                    "train_reward_accuracy": running["reward_accuracy"] / running_count,
                    "train_reward_margin": running["reward_margin"] / running_count,
                    "grad_norm": float(grad_norm),
                    "elapsed": time.time() - start,
                }
                append_jsonl(metrics_path, row)
                print(
                    f"step {step}/{total_steps} loss {row['train_loss']:.4f} "
                    f"acc {row['train_reward_accuracy']:.3f} margin {row['train_reward_margin']:.4f}",
                    flush=True,
                )
                running = {key: 0.0 for key in running}
                running_count = 0

            if step % args.eval_every == 0:
                run_eval(save_best=True)

            if step >= total_steps:
                done = True
                break
        if done:
            break

    if step % args.eval_every != 0:
        run_eval(save_best=True)
    save_checkpoint(policy, tokenizer, os.path.join(args.output_dir, "final"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
