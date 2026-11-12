"""CLI script to benchmark MHA, GQA, and MQA across context lengths."""

from __future__ import annotations

import argparse

from sciminilm.benchmarks.attention_bench import (
    benchmark_attention_variants,
    format_benchmark_table,
)
from sciminilm.config import ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MHA vs GQA vs MQA across context lengths.")
    parser.add_argument("--preset", choices=["tiny", "77m"], default="tiny")
    parser.add_argument("--contexts", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--decode-tokens", type=int, default=16)
    args = parser.parse_args()

    base_cfg = ModelConfig.sciminilm_77m() if args.preset == "77m" else ModelConfig.tiny_test()
    entries = benchmark_attention_variants(
        base_config=base_cfg,
        context_lengths=args.contexts,
        decode_tokens=args.decode_tokens,
    )
    print(format_benchmark_table(entries))


if __name__ == "__main__":
    main()
