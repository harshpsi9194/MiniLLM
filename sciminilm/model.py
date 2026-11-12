"""SciMiniLM Decoder-Only Transformer Architecture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sciminilm.config import ModelConfig
from sciminilm.inference.kv_cache import KVCache
from sciminilm.layers.attention import CausalSelfAttention
from sciminilm.layers.mlp import SwiGLUMLP
from sciminilm.layers.rmsnorm import RMSNorm


@dataclass
class ModelOutput:
    """Output container for SciMiniLM forward pass."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block with RMSNorm, CausalSelfAttention, and SwiGLU."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
            use_triton=config.use_triton_rmsnorm,
        )
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.ffn_norm = RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
            use_triton=config.use_triton_rmsnorm,
        )
        self.mlp = SwiGLUMLP(
            d_model=config.d_model,
            d_ff=config.d_ff,
            bias=config.bias,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids,
        )
        x = x + self.mlp(self.ffn_norm(x))
        return x


class SciMiniLM(nn.Module):
    """Decoder-only Transformer language model (77M-parameter preset & configurable)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
            use_triton=config.use_triton_rmsnorm,
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self._init_weights()

    def _init_weights(self) -> None:
        std = self.config.init_std
        residual_std = std / math.sqrt(2.0 * self.config.n_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                target_std = (
                    residual_std
                    if name.endswith(("o_proj", "down_proj"))
                    else std
                )
                nn.init.normal_(module.weight, mean=0.0, std=target_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def create_kv_cache(self) -> KVCache:
        """Create a fresh KVCache sized for this model's layer count."""
        return KVCache(n_layers=self.config.n_layers)

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Return unique parameter count (properly deduplicating tied weights)."""
        params = (
            (p for p in self.parameters() if p.requires_grad)
            if only_trainable
            else self.parameters()
        )
        return sum(p.numel() for p in params)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Run forward pass and optionally compute next-token cross-entropy loss.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Optional 2D (batch, k_len) or 4D attention mask.
            labels: Optional unshifted token IDs of shape (batch, seq_len). When
                provided, cross-entropy loss is computed on `logits[:, :-1]` vs
                `labels[:, 1:]` ignoring `-100`.
            kv_cache: Optional KVCache for autoregressive generation.
            position_ids: Optional explicit positions of shape (batch, seq_len).
        """
        x = self.drop(self.tok_embeddings(input_ids))
        for block in self.layers:
            x = block(
                x,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                position_ids=position_ids,
            )
        x = self.norm(x)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            if logits.shape[1] < 2:
                raise ValueError("Cannot compute shifted causal LM loss with seq_len < 2")
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return ModelOutput(logits=logits, loss=loss)
