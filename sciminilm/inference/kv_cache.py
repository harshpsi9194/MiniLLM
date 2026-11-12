"""KV-Cache management with rollback/truncation support for autoregressive and speculative decoding."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class KVCache:
    """Dynamic per-layer Key-Value cache supporting rollback for speculative decoding."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self._keys: List[Optional[torch.Tensor]] = [None] * n_layers
        self._values: List[Optional[torch.Tensor]] = [None] * n_layers

    def seq_len(self, layer_idx: int = 0) -> int:
        """Return the current cached sequence length for the specified layer."""
        k = self._keys[layer_idx]
        return 0 if k is None else int(k.shape[-2])

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new key/value tensors along the sequence dimension and return full K, V."""
        k_prev = self._keys[layer_idx]
        v_prev = self._values[layer_idx]
        if k_prev is None or v_prev is None:
            k_cat = k_new
            v_cat = v_new
        else:
            k_cat = torch.cat([k_prev, k_new], dim=-2)
            v_cat = torch.cat([v_prev, v_new], dim=-2)
        self._keys[layer_idx] = k_cat
        self._values[layer_idx] = v_cat
        return k_cat, v_cat

    def truncate(self, new_seq_len: int) -> None:
        """Truncate all layers' cached keys and values to `new_seq_len`."""
        if new_seq_len < 0:
            raise ValueError(f"new_seq_len must be non-negative, got {new_seq_len}")
        for idx in range(self.n_layers):
            k = self._keys[idx]
            v = self._values[idx]
            if k is not None and v is not None:
                if new_seq_len == 0:
                    self._keys[idx] = None
                    self._values[idx] = None
                elif k.shape[-2] > new_seq_len:
                    self._keys[idx] = k[..., :new_seq_len, :].contiguous()
                    self._values[idx] = v[..., :new_seq_len, :].contiguous()

    def clear(self) -> None:
        """Reset all cached keys and values."""
        for idx in range(self.n_layers):
            self._keys[idx] = None
            self._values[idx] = None

    def memory_bytes(self) -> int:
        """Return total memory footprint in bytes of all cached K/V tensors."""
        total = 0
        for k, v in zip(self._keys, self._values):
            if k is not None:
                total += k.numel() * k.element_size()
            if v is not None:
                total += v.numel() * v.element_size()
        return total

    @staticmethod
    def theoretical_memory_bytes(
        batch_size: int,
        seq_len: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype_bytes: int = 2,
    ) -> int:
        """Analytically compute KV-cache memory in bytes for a given configuration."""
        # 2 tensors (K and V) per layer
        return 2 * batch_size * n_layers * n_kv_heads * seq_len * head_dim * dtype_bytes
