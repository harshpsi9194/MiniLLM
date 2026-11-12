"""Benchmark suite comparing MHA, GQA, and MQA across context lengths."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from sciminilm.config import AttentionType, ModelConfig
from sciminilm.inference.engine import InferenceEngine
from sciminilm.inference.kv_cache import KVCache
from sciminilm.model import SciMiniLM
from sciminilm.quant.evaluate import evaluate_perplexity


@dataclass
class AttentionBenchmarkEntry:
    """Measured metrics for one (attention_variant, context_length) pair."""

    variant: AttentionType
    n_heads: int
    n_kv_heads: int
    context_length: int
    num_params: int
    perplexity: float
    kv_cache_bytes: int
    kv_cache_theoretical_bytes: int
    prefill_latency_ms: float
    decode_latency_ms_per_token: float
    tokens_per_second: float


def _build_variant_config(
    base_config: ModelConfig,
    variant: AttentionType,
    gqa_kv_heads: Optional[int] = None,
) -> ModelConfig:
    if variant == "mha":
        n_kv = base_config.n_heads
    elif variant == "mqa":
        n_kv = 1
    elif variant == "gqa":
        if gqa_kv_heads is not None:
            n_kv = gqa_kv_heads
        else:
            # Pick a valid intermediate divisor of n_heads
            divisors = [d for d in range(2, base_config.n_heads) if base_config.n_heads % d == 0]
            n_kv = divisors[0] if divisors else 1
    else:
        raise ValueError(f"Unknown attention variant: {variant}")

    cfg_dict = base_config.to_dict()
    cfg_dict["n_kv_heads"] = n_kv
    return ModelConfig(**cfg_dict)


@torch.no_grad()
def benchmark_attention_variants(
    base_config: ModelConfig,
    context_lengths: Sequence[int] = (64, 128, 256, 512),
    variants: Sequence[AttentionType] = ("mha", "gqa", "mqa"),
    decode_tokens: int = 16,
    batch_size: int = 1,
    gqa_kv_heads: Optional[int] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> List[AttentionBenchmarkEntry]:
    """Compare MHA, GQA, and MQA across context lengths on perplexity, KV-cache memory, and latency."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: List[AttentionBenchmarkEntry] = []

    for variant in variants:
        torch.manual_seed(seed)
        cfg = _build_variant_config(base_config, variant, gqa_kv_heads=gqa_kv_heads)
        max_needed = max(context_lengths) + decode_tokens + 4
        if cfg.max_seq_len < max_needed:
            cfg_dict = cfg.to_dict()
            cfg_dict["max_seq_len"] = max_needed
            cfg = ModelConfig(**cfg_dict)

        model = SciMiniLM(cfg).to(device)
        model.eval()
        engine = InferenceEngine(model)
        num_params = model.num_parameters()

        for ctx_len in context_lengths:
            torch.manual_seed(seed + ctx_len)
            prompt = torch.randint(
                0,
                cfg.vocab_size,
                (batch_size, ctx_len),
                dtype=torch.long,
                device=device,
            )

            # 1. Perplexity measurement on context batch
            ppl = evaluate_perplexity(model, [prompt], device=device)

            # 2. Inference latency & KV-cache memory measurement
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_start = time.perf_counter()
            gen_res = engine.generate(
                prompt,
                max_new_tokens=decode_tokens,
                do_sample=False,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            _ = time.perf_counter() - t_start

            # The final sampled token is not fed back into the model, so KV cache holds
            # prompt + (num_generated_tokens - 1) tokens.
            total_kv_len = ctx_len + max(0, gen_res.num_generated_tokens - 1)
            dtype_bytes = next(model.parameters()).element_size()
            theoretical_bytes = KVCache.theoretical_memory_bytes(
                batch_size=batch_size,
                seq_len=total_kv_len,
                n_layers=cfg.n_layers,
                n_kv_heads=cfg.n_kv_heads or cfg.n_heads,
                head_dim=cfg.head_dim,
                dtype_bytes=dtype_bytes,
            )

            decode_ms_per_tok = (
                (gen_res.decode_time_s * 1000.0) / max(1, gen_res.num_generated_tokens)
            )
            results.append(
                AttentionBenchmarkEntry(
                    variant=variant,
                    n_heads=cfg.n_heads,
                    n_kv_heads=cfg.n_kv_heads or cfg.n_heads,
                    context_length=ctx_len,
                    num_params=num_params,
                    perplexity=ppl,
                    kv_cache_bytes=gen_res.kv_cache_bytes,
                    kv_cache_theoretical_bytes=theoretical_bytes,
                    prefill_latency_ms=gen_res.prefill_time_s * 1000.0,
                    decode_latency_ms_per_token=decode_ms_per_tok,
                    tokens_per_second=gen_res.tokens_per_second,
                )
            )

    return results


def format_benchmark_table(entries: Sequence[AttentionBenchmarkEntry]) -> str:
    """Render benchmark entries as a human-readable Markdown/ASCII summary table."""
    header = (
        f"{'Variant':<8} | {'KV Heads':<8} | {'Ctx Len':<8} | {'Params':<10} | "
        f"{'PPL':<10} | {'KV Mem (KB)':<11} | {'Prefill (ms)':<12} | {'Decode (ms/tok)':<15}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for e in entries:
        lines.append(
            f"{e.variant.upper():<8} | {e.n_kv_heads:<8} | {e.context_length:<8} | "
            f"{e.num_params:<10} | {e.perplexity:<10.3f} | {e.kv_cache_bytes / 1024.0:<11.2f} | "
            f"{e.prefill_latency_ms:<12.2f} | {e.decode_latency_ms_per_token:<15.3f}"
        )
    return "\n".join(lines)
