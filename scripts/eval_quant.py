"""CLI script to evaluate INT8/INT4 quantization memory reduction and perplexity."""

from __future__ import annotations

import argparse
import torch

from sciminilm.config import ModelConfig, QuantConfig
from sciminilm.model import SciMiniLM
from sciminilm.quant.evaluate import evaluate_quantization


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate INT8/INT4 weight quantization.")
    parser.add_argument("--preset", choices=["tiny", "77m"], default="tiny")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=4)
    args = parser.parse_args()

    cfg = ModelConfig.sciminilm_77m() if args.preset == "77m" else ModelConfig.tiny_test()
    model = SciMiniLM(cfg)
    batches = [
        torch.randint(0, cfg.vocab_size, (2, args.seq_len), dtype=torch.long)
        for _ in range(args.num_batches)
    ]
    results = evaluate_quantization(
        model,
        batches,
        configs={
            "int8_per_channel": QuantConfig(bits=8, symmetric=True),
            "int4_g64_linear": QuantConfig(bits=4, group_size=64, symmetric=True),
            "int4_g64_full": QuantConfig(
                bits=4, group_size=64, symmetric=True, quantize_embeddings=True
            ),
        },
    )
    for res in results:
        print(
            f"{res.name:<18} | PPL={res.perplexity:8.3f} ({res.ppl_degradation_pct:+6.2f}%) | "
            f"Mem={res.memory.total_mb:6.2f} MB | "
            f"vsFP32={res.memory.reduction_vs_fp32:4.2f}x | "
            f"LinearVsFP16={res.memory.linear_reduction_vs_fp16:4.2f}x"
        )


if __name__ == "__main__":
    main()
