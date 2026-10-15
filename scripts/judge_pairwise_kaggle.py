import argparse
import hashlib
import json
import os

from alignment.judging import judge_batches, summarize
from alignment.utils import append_jsonl, read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--judge_model", default="google/gemini-3.6-flash")
    parser.add_argument("--backend", choices=["kaggle", "mock"], default="kaggle")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=float, default=5.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--reuse_judgments", action="store_true")
    parser.add_argument("--subject_name", default="dpo")
    parser.add_argument("--baseline_name", default="sft")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def kaggle_caller(model_name):
    import kaggle_benchmarks as kbench

    try:
        llm = kbench.llms[model_name]
    except Exception:
        llm = kbench.llms[model_name.split("/")[-1]]
    holder = {}

    @kbench.task(name="pairwise_judge_call")
    def judge_call(llm, prompt: str):
        holder["text"] = str(llm.prompt(prompt))

    def call(prompt):
        holder.pop("text", None)
        judge_call.run(llm=llm, prompt=prompt)
        if "text" not in holder:
            raise RuntimeError("judge task produced no response")
        return holder["text"]

    return call


def mock_caller():
    def call(prompt):
        ids = [line[len("### Pair ") :].strip() for line in prompt.splitlines() if line.startswith("### Pair ")]
        picks = ["A", "B", "tie"]
        items = [
            {"pair_id": i, "winner": picks[int(hashlib.md5(i.encode()).hexdigest(), 16) % 3]} for i in ids
        ]
        return json.dumps(items)

    return call


def main():
    args = parse_args()
    if args.backend == "kaggle" and args.sleep_seconds < 5:
        raise SystemExit("the manual requires at least 5 seconds between judge request attempts")

    pairs = read_jsonl(args.pairs_path)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    already = {}
    if args.reuse_judgments and os.path.exists(args.output_path):
        already = {row["pair_id"]: row for row in read_jsonl(args.output_path)}
    elif os.path.exists(args.output_path):
        os.remove(args.output_path)
    pending = [row for row in pairs if row["pair_id"] not in already]
    print(f"{len(already)} reused, {len(pending)} to judge", flush=True)

    failure_path = args.output_path + ".failures.jsonl"
    call_model = kaggle_caller(args.judge_model) if args.backend == "kaggle" else mock_caller()

    def on_judged(rows):
        for row in rows:
            append_jsonl(args.output_path, row)
            already[row["pair_id"]] = row
        print(f"judged {len(already)}/{len(pairs)}", flush=True)

    def on_failure(row):
        append_jsonl(failure_path, row)

    stats = judge_batches(
        pending,
        call_model,
        args.batch_size,
        args.sleep_seconds if args.backend == "kaggle" else 0.0,
        args.max_retries,
        args.judge_model,
        on_judged,
        on_failure,
    )

    judged = [already[row["pair_id"]] for row in pairs if row["pair_id"] in already]
    unresolved = len(pairs) - len(judged)
    summary = summarize(
        judged,
        args.subject_name,
        args.baseline_name,
        failures=unresolved,
        retries=stats.get("retries", 0),
        judge_model=args.judge_model,
    )
    with open(args.output_path + ".summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=2))


if __name__ == "__main__":
    main()
