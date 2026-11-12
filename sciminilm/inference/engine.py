"""Autoregressive inference engine with KV caching and Top-K/Top-P sampling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch

from sciminilm.config import InferenceConfig
from sciminilm.inference.kv_cache import KVCache
from sciminilm.inference.sampling import sample_next_token
from sciminilm.model import SciMiniLM


@dataclass
class GenerationResult:
    """Output and performance metrics from autoregressive or speculative generation."""

    sequences: torch.Tensor
    generated_tokens: torch.Tensor
    prefill_time_s: float
    decode_time_s: float
    total_time_s: float
    num_generated_tokens: int
    tokens_per_second: float
    kv_cache_bytes: int


class InferenceEngine:
    """KV-cached autoregressive generation engine for SciMiniLM."""

    def __init__(self, model: SciMiniLM) -> None:
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        config: Optional[InferenceConfig] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        eos_token_id: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> GenerationResult:
        """Generate continuation tokens given `prompt_ids` of shape (batch, prompt_len)."""
        if prompt_ids.ndim != 2:
            raise ValueError(f"Expected 2D prompt_ids (batch, seq_len), got {prompt_ids.shape}")
        if prompt_ids.shape[1] == 0:
            raise ValueError("prompt_ids must contain at least 1 token")

        cfg = config or InferenceConfig()
        max_new = cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        temp = cfg.temperature if temperature is None else temperature
        k_val = cfg.top_k if top_k is None else top_k
        p_val = cfg.top_p if top_p is None else top_p
        sample_flag = cfg.do_sample if do_sample is None else do_sample
        eos_id = cfg.eos_token_id if eos_token_id is None else eos_token_id

        was_training = self.model.training
        self.model.eval()
        device = next(self.model.parameters()).device
        prompt_ids = prompt_ids.to(device)

        kv_cache: KVCache = self.model.create_kv_cache()
        batch_size, prompt_len = prompt_ids.shape
        max_allowed_new = min(max_new, max(0, self.model.config.max_seq_len - prompt_len))

        t0 = time.perf_counter()
        out = self.model(prompt_ids, kv_cache=kv_cache)
        t_prefill_end = time.perf_counter()

        if max_allowed_new == 0:
            if was_training:
                self.model.train()
            empty_gen = torch.empty((batch_size, 0), dtype=prompt_ids.dtype, device=device)
            return GenerationResult(
                sequences=prompt_ids,
                generated_tokens=empty_gen,
                prefill_time_s=t_prefill_end - t0,
                decode_time_s=0.0,
                total_time_s=t_prefill_end - t0,
                num_generated_tokens=0,
                tokens_per_second=0.0,
                kv_cache_bytes=kv_cache.memory_bytes(),
            )

        next_token = sample_next_token(
            out.logits[:, -1, :],
            temperature=temp,
            top_k=k_val,
            top_p=p_val,
            do_sample=sample_flag,
            generator=generator,
        )

        generated_list = [next_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if eos_id is not None:
            finished |= next_token == eos_id

        t_decode_start = time.perf_counter()
        for _ in range(1, max_allowed_new):
            if bool(finished.all()):
                break
            step_input = next_token.unsqueeze(1)
            step_out = self.model(step_input, kv_cache=kv_cache)
            next_token = sample_next_token(
                step_out.logits[:, -1, :],
                temperature=temp,
                top_k=k_val,
                top_p=p_val,
                do_sample=sample_flag,
                generator=generator,
            )
            if eos_id is not None:
                next_token = torch.where(finished, eos_id, next_token)
                finished |= next_token == eos_id
            generated_list.append(next_token)
        t_decode_end = time.perf_counter()

        if was_training:
            self.model.train()

        generated_tokens = torch.stack(generated_list, dim=1)
        sequences = torch.cat([prompt_ids, generated_tokens], dim=1)

        n_gen = generated_tokens.shape[1]
        prefill_s = t_prefill_end - t0
        decode_s = t_decode_end - t_decode_start
        total_s = t_decode_end - t0
        tps = (n_gen * batch_size) / max(decode_s, 1e-9) if n_gen > 1 else n_gen / max(total_s, 1e-9)

        return GenerationResult(
            sequences=sequences,
            generated_tokens=generated_tokens,
            prefill_time_s=prefill_s,
            decode_time_s=decode_s,
            total_time_s=total_s,
            num_generated_tokens=n_gen,
            tokens_per_second=tps,
            kv_cache_bytes=kv_cache.memory_bytes(),
        )
