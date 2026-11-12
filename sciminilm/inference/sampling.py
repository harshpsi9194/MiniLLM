"""Sampling utilities (greedy, temperature, Top-K, Top-P nucleus sampling)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask logits outside the top-k highest values per row to -inf."""
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    kth_vals = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_vals, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Mask logits outside the nucleus cumulative probability `top_p` to -inf."""
    if top_p >= 1.0 or top_p <= 0.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits.float(), dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold, keeping at least 1 token
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))


def compute_sampling_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Convert raw logits (..., vocab) into filtered sampling probabilities."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0 for probability computation, got {temperature}")
    scaled = logits.float() / temperature
    if top_k > 0:
        scaled = apply_top_k(scaled, top_k)
    if 0.0 < top_p < 1.0:
        scaled = apply_top_p(scaled, top_p)
    return F.softmax(scaled, dim=-1)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    do_sample: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample next token IDs from logits of shape (batch, vocab_size)."""
    if not do_sample or temperature <= 0.0:
        return torch.argmax(logits, dim=-1)
    probs = compute_sampling_probs(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
