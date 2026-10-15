import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def smooth(values, window):
    out = []
    for index in range(len(values)):
        chunk = values[max(0, index - window + 1) : index + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def rows_of(rows, kind):
    return [row for row in rows if row.get("type") == kind]


def plot_sft(metrics_path, output_path, window):
    rows = read_jsonl(metrics_path)
    train, evals = rows_of(rows, "train"), rows_of(rows, "eval")
    best = min(evals, key=lambda row: row["val_loss"])
    fig, axis = plt.subplots(figsize=(6, 3.6))
    steps = [row["step"] for row in train]
    losses = [row["train_loss"] for row in train]
    axis.plot(steps, losses, color="tab:blue", alpha=0.25, label="train loss (raw, per 10 steps)")
    axis.plot(steps, smooth(losses, window), color="tab:blue", label=f"train loss (moving avg, {window} points)")
    axis.plot(
        [row["step"] for row in evals],
        [row["val_loss"] for row in evals],
        color="tab:orange",
        marker="o",
        label="validation loss (every 100 steps)",
    )
    axis.scatter([best["step"]], [best["val_loss"]], color="red", zorder=5, label=f"best checkpoint (step {best['step']})")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("cross-entropy loss (response tokens)")
    axis.set_title("SFT: Qwen2.5-0.5B on 30k UltraChat examples")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_dpo(metrics_path, output_path, title, window):
    rows = read_jsonl(metrics_path)
    train, evals, bests = rows_of(rows, "train"), rows_of(rows, "eval"), rows_of(rows, "best")
    best_step = bests[-1]["step"] if bests else None
    eval_steps = [row["step"] for row in evals]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    steps = [row["step"] for row in train]
    losses = [row["train_loss"] for row in train]
    axes[0].plot(steps, losses, color="tab:blue", alpha=0.25, label="train loss (raw)")
    axes[0].plot(steps, smooth(losses, window), color="tab:blue", label=f"train loss (moving avg, {window})")
    axes[0].plot(eval_steps, [row["val_loss"] for row in evals], color="tab:orange", marker="o", label="validation loss")
    axes[0].set_ylabel("DPO loss")

    accs = [row["train_reward_accuracy"] for row in train]
    axes[1].plot(steps, accs, color="tab:blue", alpha=0.25, label="train accuracy (raw)")
    axes[1].plot(steps, smooth(accs, window), color="tab:blue", label=f"train accuracy (moving avg, {window})")
    axes[1].plot(eval_steps, [row["reward_accuracy"] for row in evals], color="tab:orange", marker="o", label="validation accuracy")
    axes[1].axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axes[1].set_ylabel("reward accuracy")

    margins = [row["train_reward_margin"] for row in train]
    axes[2].plot(steps, margins, color="tab:blue", alpha=0.25, label="train margin (raw)")
    axes[2].plot(steps, smooth(margins, window), color="tab:blue", label=f"train margin (moving avg, {window})")
    axes[2].plot(eval_steps, [row["reward_margin"] for row in evals], color="tab:orange", marker="o", label="validation margin")
    axes[2].set_ylabel("reward margin")

    for axis in axes:
        axis.set_xlabel("optimizer step")
        if best_step is not None:
            axis.axvline(best_step, color="red", linestyle="--", linewidth=1, label=f"best checkpoint (step {best_step})")
        axis.legend(fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_lora(metrics_path, output_path):
    with open(metrics_path, "r", encoding="utf-8") as handle:
        history = json.load(handle)["log_history"]
    train = [row for row in history if "loss" in row and "eval_loss" not in row]
    evals = [row for row in history if "eval_loss" in row]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot([r["step"] for r in train], [r["loss"] for r in train], marker=".", label="train loss")
    axes[0].plot([r["step"] for r in evals], [r["eval_loss"] for r in evals], marker="o", label="validation loss")
    axes[0].set_ylabel("DPO loss")
    if train and "rewards/accuracies" in train[0]:
        axes[1].plot([r["step"] for r in train], [r["rewards/accuracies"] for r in train], marker=".", label="train accuracy")
    if evals and "eval_rewards/accuracies" in evals[0]:
        axes[1].plot([r["step"] for r in evals], [r["eval_rewards/accuracies"] for r in evals], marker="o", label="validation accuracy")
    axes[1].set_ylabel("reward accuracy")
    if train and "rewards/margins" in train[0]:
        axes[2].plot([r["step"] for r in train], [r["rewards/margins"] for r in train], marker=".", label="train margin")
    if evals and "eval_rewards/margins" in evals[0]:
        axes[2].plot([r["step"] for r in evals], [r["eval_rewards/margins"] for r in evals], marker="o", label="validation margin")
    axes[2].set_ylabel("reward margin")
    for axis in axes:
        axis.set_xlabel("optimizer step")
        axis.legend(fontsize=7)
    fig.suptitle("LoRA-DPO: SmolLM2-360M-Instruct on persona preferences")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--plots_dir", default="outputs/plots")
    parser.add_argument("--window", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)
    jobs = [
        ("sft", lambda p: plot_sft(f"{args.outputs_dir}/sft/metrics.jsonl", p, args.window), f"{args.outputs_dir}/sft/metrics.jsonl"),
        (
            "dpo_hh",
            lambda p: plot_dpo(f"{args.outputs_dir}/dpo_hh/metrics.jsonl", p, "DPO on HH-RLHF (15k pairs, beta 0.1, lr 1e-6)", args.window),
            f"{args.outputs_dir}/dpo_hh/metrics.jsonl",
        ),
        (
            "dpo_ultrafeedback",
            lambda p: plot_dpo(
                f"{args.outputs_dir}/dpo_ultrafeedback/metrics.jsonl", p, "DPO on UltraFeedback (beta 0.1, lr 1e-6)", args.window
            ),
            f"{args.outputs_dir}/dpo_ultrafeedback/metrics.jsonl",
        ),
        ("lora_dpo", lambda p: plot_lora(f"{args.outputs_dir}/lora_dpo/metrics.json", p), f"{args.outputs_dir}/lora_dpo/metrics.json"),
    ]
    for name, draw, source in jobs:
        if not os.path.exists(source):
            print(f"skip {name}: {source} not found")
            continue
        target = os.path.join(args.plots_dir, f"{name}.png")
        draw(target)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
