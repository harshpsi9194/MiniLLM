"""Benchmarking utilities for SciMiniLM."""

from sciminilm.benchmarks.attention_bench import (
    AttentionBenchmarkEntry,
    benchmark_attention_variants,
    format_benchmark_table,
)

__all__ = [
    "AttentionBenchmarkEntry",
    "benchmark_attention_variants",
    "format_benchmark_table",
]
