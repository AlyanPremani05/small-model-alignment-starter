from __future__ import annotations

from types import SimpleNamespace

import math
import torch
from torch import nn

from alignment import core
from tests.toy_lm import ToyTokenizer


class BigramLM(nn.Module):
    def __init__(self, vocab_size, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.table = nn.Parameter(torch.randn(vocab_size, vocab_size, generator=g))

    def forward(self, input_ids, attention_mask=None, labels=None):
        return SimpleNamespace(logits=self.table[input_ids])


def manual_logp(model, tok, prompt, response, max_length):
    p = tok.encode(prompt)
    r = tok.encode(response) + [tok.eos_token_id]
    ids = (p + r)[:max_length]
    n_prompt = len(p)
    total = 0.0
    for t in range(max(n_prompt, 1), len(ids)):
        logp = torch.log_softmax(model.table[ids[t - 1]], dim=-1)[ids[t]]
        total += logp.item()
    return total


def test_logprobs_match_manual_bigram_computation():
    tok = ToyTokenizer()
    model = BigramLM(tok.vocab_size)
    prompts = ["Q: hi ", "A: "]
    responses = ["there", "a much longer response"]
    got = core.response_logprobs(model, tok, prompts, responses, max_length=64)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        assert math.isclose(got[i].item(), manual_logp(model, tok, p, r, 64), rel_tol=1e-5)


def test_logprobs_independent_of_batch_composition_and_order():
    tok = ToyTokenizer()
    model = BigramLM(tok.vocab_size)
    prompts = ["Q: ", "Longer prompt here: ", "x "]
    responses = ["a", "bcd efg", "hello"]
    together = core.response_logprobs(model, tok, prompts, responses, 64)
    alone = torch.stack([core.response_logprobs(model, tok, [p], [r], 64)[0] for p, r in zip(prompts, responses)])
    torch.testing.assert_close(together, alone)
    reversed_ = core.response_logprobs(model, tok, prompts[::-1], responses[::-1], 64)
    torch.testing.assert_close(reversed_.flip(0), together)


def test_truncation_drops_eos_and_pads_to_longest_not_max_length():
    tok = ToyTokenizer()
    batch = core.build_lm_batch(tok, ["ab"], ["cdefgh"], max_length=5)
    assert batch["input_ids"].shape == (1, 5)
    assert tok.eos_token_id not in batch["input_ids"][0].tolist()
    assert batch["labels"][0].tolist()[:2] == [-100, -100]

    short = core.build_lm_batch(tok, ["ab"], ["c"], max_length=50)
    assert short["input_ids"].shape == (1, 4)
    assert short["input_ids"][0, -1] == tok.eos_token_id


def test_prompt_longer_than_max_length_gives_zero_logp_and_finite_loss():
    tok = ToyTokenizer()
    model = BigramLM(tok.vocab_size)
    lp = core.response_logprobs(model, tok, ["a" * 30], ["b"], max_length=10)
    assert lp.item() == 0.0
    loss = core.sft_loss(model, tok, ["a" * 30], ["b"], max_length=10)
    assert torch.isfinite(loss) and loss.item() == 0.0


def test_dpo_loss_from_logps_matches_formula():
    pc, pr = torch.tensor([-3.0, -4.0]), torch.tensor([-5.0, -3.5])
    rc, rr = torch.tensor([-3.5, -4.2]), torch.tensor([-4.5, -3.4])
    beta = 0.1
    z = beta * ((pc - pr) - (rc - rr))
    expected = (-torch.log(torch.sigmoid(z))).mean()
    torch.testing.assert_close(core.dpo_loss_from_logps(pc, pr, rc, rr, beta), expected)


def test_dpo_loss_decreases_when_policy_favors_chosen_and_ref_gets_no_grad():
    tok = ToyTokenizer()
    policy = BigramLM(tok.vocab_size, seed=1)
    reference = BigramLM(tok.vocab_size, seed=1)
    args = (tok, ["Pick: "] * 2, ["good", "yes"], ["bad", "no"])
    loss0 = core.dpo_loss(policy, reference, *args, beta=0.1, max_length=32)
    torch.testing.assert_close(loss0, torch.log(torch.tensor(2.0)))
    loss0.backward()
    assert reference.table.grad is None
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    opt.step()
    loss1 = core.dpo_loss(policy, reference, *args, beta=0.1, max_length=32)
    assert loss1 < loss0


def test_reward_stats_start_at_zero_accuracy_when_policy_equals_reference():
    x = torch.tensor([-2.0, -3.0])
    stats = core.dpo_reward_stats(x, x - 1, x, x - 1, beta=0.1)
    assert stats["reward_accuracy"] == 0.0 and stats["reward_margin"] == 0.0
