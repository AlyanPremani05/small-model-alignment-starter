from __future__ import annotations
import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _pad_id(tokenizer) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return 0 if pad is None else pad


def _encode_row(tokenizer, prompt: str, response: str, max_length: int):
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    response_ids = list(tokenizer.encode(response, add_special_tokens=False))
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        response_ids.append(eos)
    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def build_lm_batch(tokenizer, prompts, responses, max_length):
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    rows = [_encode_row(tokenizer, p, r, max_length) for p, r in zip(prompts, responses)]
    width = max((len(ids) for ids, _ in rows), default=0)
    pad = _pad_id(tokenizer)

    input_ids, attention_mask, labels = [], [], []
    for ids, lab in rows:
        n_pad = width - len(ids)
        input_ids.append(ids + [pad] * n_pad)
        attention_mask.append([1] * len(ids) + [0] * n_pad)
        labels.append(lab + [IGNORE_INDEX] * n_pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen and rejected must have the same length")
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_lm_batch(tokenizer, prompts, rejected_responses, max_length),
    }


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def sequence_logps(model, batch):
    device = _model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)
    safe_targets = targets.masked_fill(~mask, 0)

    token_logps = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    logps = (token_logps * mask).sum(dim=-1)
    return logps, mask.sum(dim=-1)


def response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_lm_batch(tokenizer, prompts, responses, max_length)
    logps, _ = sequence_logps(model, batch)
    return logps


def sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_lm_batch(tokenizer, prompts, responses, max_length)
    return sft_loss_from_batch(model, batch)


def sft_loss_from_batch(model, batch):
    logps, counts = sequence_logps(model, batch)
    return -logps.sum() / counts.sum().clamp_min(1)


def dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    return -F.logsigmoid(beta * (policy_gap - reference_gap)).mean()


@torch.no_grad()
def dpo_reward_stats(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    r_w = beta * (policy_chosen_logps - reference_chosen_logps)
    r_l = beta * (policy_rejected_logps - reference_rejected_logps)
    return {
        "reward_accuracy": (r_w > r_l).float().mean().item(),
        "reward_margin": (r_w - r_l).mean().item(),
        "reward_chosen": r_w.mean().item(),
        "reward_rejected": r_l.mean().item(),
    }


def dpo_loss_from_batch(policy_model, reference_model, batch, beta):
    pol_c, _ = sequence_logps(policy_model, batch["chosen"])
    pol_r, _ = sequence_logps(policy_model, batch["rejected"])
    with torch.no_grad():
        ref_c, _ = sequence_logps(reference_model, batch["chosen"])
        ref_r, _ = sequence_logps(reference_model, batch["rejected"])
    ref_c, ref_r = ref_c.to(pol_c.device), ref_r.to(pol_c.device)
    loss = dpo_loss_from_logps(pol_c, pol_r, ref_c, ref_r, beta)
    stats = dpo_reward_stats(pol_c, pol_r, ref_c, ref_r, beta)
    return loss, stats


def dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    loss, _ = dpo_loss_from_batch(policy_model, reference_model, batch, beta)
    return loss
