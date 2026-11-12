"""Configurable MHA, GQA, and MQA Causal Self-Attention with FlashAttention/SDPA."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sciminilm.config import ModelConfig
from sciminilm.layers.rope import RotaryEmbedding

if TYPE_CHECKING:
    from sciminilm.inference.kv_cache import KVCache


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match the number of query heads.

    Args:
        x: Tensor of shape (batch, n_kv_heads, seq_len, head_dim).
        n_rep: Repetition factor (n_heads // n_kv_heads).

    Returns:
        Tensor of shape (batch, n_kv_heads * n_rep, seq_len, head_dim).
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x.unsqueeze(2)
        .expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)
    )


def build_causal_mask(
    q_len: int,
    k_len: int,
    device: torch.device,
    past_len: int = 0,
) -> torch.Tensor:
    """Build a boolean causal mask of shape (1, 1, q_len, k_len) where True means attend."""
    mask = torch.tril(
        torch.ones((q_len, k_len), dtype=torch.bool, device=device),
        diagonal=past_len,
    )
    return mask.unsqueeze(0).unsqueeze(0)


def _manual_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
) -> torch.Tensor:
    """Pure-PyTorch manual attention fallback when FlashAttention/SDPA is disabled."""
    q_len = q.shape[-2]
    k_len = k.shape[-2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_causal:
        past_len = k_len - q_len
        causal_mask = build_causal_mask(q_len, k_len, q.device, past_len=past_len)
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
    if dropout_p > 0.0:
        probs = F.dropout(probs, p=dropout_p, training=True)
    return torch.matmul(probs, v)


class CausalSelfAttention(nn.Module):
    """Multi-Head (MHA), Grouped-Query (GQA), or Multi-Query (MQA) causal attention."""

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        assert config.n_kv_heads is not None
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout
        self.use_flash_attention = config.use_flash_attention

        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, kv_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=config.bias)

        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.resid_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute causal self-attention with RoPE and optional KV caching.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            attention_mask: Optional mask. Either 2D padding mask (batch, total_k_len)
                with 1/True for valid tokens and 0/False for masked tokens, or a 4D
                mask broadcastable to (batch, n_heads, q_len, total_k_len).
            kv_cache: Optional KVCache instance for autoregressive decoding.
            position_ids: Optional explicit position indices of shape (batch, seq_len).
        """
        batch, q_len, _ = x.shape

        q = self.q_proj(x).view(batch, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = kv_cache.seq_len(self.layer_idx) if kv_cache is not None else 0
        q, k = self.rope(q, k, position_offset=past_len, position_ids=position_ids)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        k_len = k.shape[-2]
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        dropout_p = self.dropout_p if self.training else 0.0

        # Prepare mask and causality flag
        resolved_mask: Optional[torch.Tensor] = None
        use_is_causal = False

        if attention_mask is None:
            if q_len == k_len:
                use_is_causal = True
            elif q_len == 1:
                use_is_causal = False
            else:
                # Chunked prefill or speculative verification: q_len > 1 with past_len > 0
                resolved_mask = build_causal_mask(q_len, k_len, q.device, past_len=past_len)
        else:
            causal_mask = build_causal_mask(q_len, k_len, q.device, past_len=past_len)
            if attention_mask.ndim == 2:
                # (batch, k_len) -> (batch, 1, 1, k_len)
                pad_mask = attention_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)
                resolved_mask = causal_mask & pad_mask
            elif attention_mask.ndim == 4:
                if attention_mask.dtype == torch.bool:
                    resolved_mask = causal_mask & attention_mask
                else:
                    # Additive float mask
                    additive_causal = torch.zeros(
                        (1, 1, q_len, k_len), dtype=q.dtype, device=q.device
                    )
                    additive_causal = additive_causal.masked_fill(~causal_mask, float("-inf"))
                    resolved_mask = additive_causal + attention_mask.to(dtype=q.dtype)
            else:
                raise ValueError(
                    f"Unsupported attention_mask ndim={attention_mask.ndim}; expected 2 or 4"
                )

        if self.use_flash_attention:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=resolved_mask,
                dropout_p=dropout_p,
                is_causal=use_is_causal,
            )
        else:
            attn_out = _manual_scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=resolved_mask,
                dropout_p=dropout_p,
                is_causal=use_is_causal,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, q_len, self.d_model)
        return self.resid_dropout(self.o_proj(attn_out))
