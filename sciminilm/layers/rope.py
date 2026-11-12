"""Rotary Position Embeddings (RoPE) for causal attention."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the negative first half."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply precomputed cos/sin rotary embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, head_dim).
        k: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim).
        cos: Cosine tensor broadcastable to (batch, 1, seq_len, head_dim).
        sin: Sine tensor broadcastable to (batch, 1, seq_len, head_dim).
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Precomputed Rotary Position Embedding (RoPE) cache."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len, device=inv_freq.device)

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to q and k at the given sequence positions."""
        seq_len = q.shape[-2]
        if position_ids is not None:
            needed_len = int(position_ids.max().item()) + 1
        else:
            needed_len = position_offset + seq_len

        if needed_len > self.cos_cached.shape[0] or self.cos_cached.device != q.device:
            self._build_cache(max(needed_len, self.max_seq_len), device=q.device)

        if position_ids is not None:
            # position_ids: (batch, seq_len) -> (batch, 1, seq_len, dim)
            cos = self.cos_cached[position_ids].unsqueeze(1).to(dtype=q.dtype)
            sin = self.sin_cached[position_ids].unsqueeze(1).to(dtype=q.dtype)
        else:
            cos = (
                self.cos_cached[position_offset : position_offset + seq_len]
                .unsqueeze(0)
                .unsqueeze(0)
                .to(dtype=q.dtype)
            )
            sin = (
                self.sin_cached[position_offset : position_offset + seq_len]
                .unsqueeze(0)
                .unsqueeze(0)
                .to(dtype=q.dtype)
            )

        return apply_rotary_pos_emb(q, k, cos, sin)
