"""Tests for MHA vs GQA vs MQA benchmarking suite."""

import torch

from sciminilm.benchmarks.attention_bench import (
    benchmark_attention_variants,
    format_benchmark_table,
)
from sciminilm.config import ModelConfig


def test_attention_benchmark_across_variants_and_contexts():
    cfg = ModelConfig.tiny_test(max_seq_len=128)
    entries = benchmark_attention_variants(
        base_config=cfg,
        context_lengths=(16, 32),
        variants=("mha", "gqa", "mqa"),
        decode_tokens=4,
        device=torch.device("cpu"),
    )
    assert len(entries) == 6

    by_key = {(e.variant, e.context_length): e for e in entries}
    # KV-cache memory at same context length must follow MHA > GQA > MQA
    assert by_key[("mha", 32)].kv_cache_bytes > by_key[("gqa", 32)].kv_cache_bytes
    assert by_key[("gqa", 32)].kv_cache_bytes > by_key[("mqa", 32)].kv_cache_bytes

    # Longer context uses more KV-cache memory
    assert by_key[("gqa", 32)].kv_cache_bytes > by_key[("gqa", 16)].kv_cache_bytes
    assert by_key[("gqa", 32)].kv_cache_bytes == by_key[("gqa", 32)].kv_cache_theoretical_bytes

    table = format_benchmark_table(entries)
    assert "MHA" in table and "GQA" in table and "MQA" in table
