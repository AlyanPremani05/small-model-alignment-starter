import json

from alignment.judging import build_prompt, judge_batches, parse_response, summarize


def make_pair(i, a_model="dpo", b_model="sft"):
    return {
        "pair_id": f"p{i}",
        "instruction": f"inst {i}",
        "response_a": f"ra {i}",
        "response_b": f"rb {i}",
        "kind": "general",
        "category": "c1" if i % 2 else "c2",
        "hidden": {"a_model": a_model, "b_model": b_model},
    }


def run(pairs, caller, max_retries=3, batch_size=5):
    judged, failures, sleeps = [], [], []
    stats = judge_batches(
        pairs,
        caller,
        batch_size,
        5.0,
        max_retries,
        "judge",
        lambda rows: judged.extend(rows),
        lambda row: failures.append(row),
        sleep_fn=sleeps.append,
    )
    return judged, failures, sleeps, stats


def test_prompt_never_leaks_model_identity():
    pairs = [make_pair(i) for i in range(3)]
    prompt = build_prompt(pairs, "general")
    assert "dpo" not in prompt.lower() and "sft" not in prompt.lower() and "hidden" not in prompt
    assert "Response A" in prompt and "Response B" in prompt


def test_safety_prompt_adds_refusal_criteria():
    assert "refusal" in build_prompt([make_pair(0)], "safety")
    assert "refusal" not in build_prompt([make_pair(0)], "general")


def test_persona_prompt_uses_style_criteria_and_hides_identity():
    prompt = build_prompt([make_pair(0)], "persona")
    assert "persona" in prompt and "sft" not in prompt.lower() and "hidden" not in prompt


def test_parse_handles_fences_garbage_and_invalid_values():
    text = 'Sure:\n```json\n[{"pair_id": "p0", "winner": "a"}, {"pair_id": "p1", "winner": "Tie"}, {"pair_id": "p2", "winner": "maybe"}, {"pair_id": "zzz", "winner": "A"}]\n```'
    assert parse_response(text, ["p0", "p1", "p2"]) == {"p0": "A", "p1": "tie", "p2": "unknown"}
    assert parse_response("no json here", ["p0"]) == {}
    assert parse_response("[not valid json]", ["p0"]) == {}


def test_winner_is_mapped_back_through_hidden_metadata():
    pairs = [make_pair(0, "dpo", "sft"), make_pair(1, "sft", "dpo")]
    caller = lambda prompt: json.dumps([{"pair_id": "p0", "winner": "A"}, {"pair_id": "p1", "winner": "A"}])
    judged, failures, _, _ = run(pairs, caller)
    assert [row["winner_model"] for row in judged] == ["dpo", "sft"]
    assert not failures


def test_retries_only_unresolved_pairs_and_sleeps_between_attempts():
    pairs = [make_pair(i) for i in range(3)]
    calls = []

    def caller(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps([{"pair_id": "p0", "winner": "A"}])
        return json.dumps([{"pair_id": "p1", "winner": "B"}, {"pair_id": "p2", "winner": "tie"}])

    judged, failures, sleeps, stats = run(pairs, caller)
    assert len(judged) == 3 and not failures
    assert "### Pair p0" not in calls[1] and "### Pair p1" in calls[1]
    assert sleeps == [5.0] and stats["retries"] == 1


def test_exceptions_and_exhausted_retries_become_failures_not_crashes():
    pairs = [make_pair(0)]

    def caller(prompt):
        raise RuntimeError("denied")

    judged, failures, sleeps, stats = run(pairs, caller, max_retries=3)
    assert judged == [] and len(failures) == 1
    assert sleeps == [5.0, 5.0] and stats["requests"] == 3


def test_persistently_invalid_winner_is_recorded_as_unknown():
    caller = lambda prompt: json.dumps([{"pair_id": "p0", "winner": "neither"}])
    judged, failures, _, _ = run([make_pair(0)], caller, max_retries=2)
    assert judged[0]["winner_model"] == "unknown" and not failures


def test_summary_counts_and_rates():
    rows = []
    for i, decision in enumerate(["A", "A", "B", "tie", "A"]):
        pair = make_pair(i)
        pair["winner_model"] = {"A": "dpo", "B": "sft", "tie": "tie"}[decision]
        rows.append(pair)
    summary = summarize(rows, "dpo", "sft", failures=1, retries=2, judge_model="j")
    assert (summary["wins"], summary["losses"], summary["ties"]) == (3, 1, 1)
    assert summary["win_rate_excluding_ties"] == 0.75
    assert summary["win_rate_ties_as_half"] == 3.5 / 5
    assert set(summary["by_category"]) == {"c1", "c2"}
