"""CLI script demonstrating KV-cached autoregressive generation & speculative decoding."""

from __future__ import annotations

import argparse
import torch

from sciminilm.config import ModelConfig
from sciminilm.inference.engine import InferenceEngine
from sciminilm.inference.speculative import SpeculativeDecoder
from sciminilm.model import SciMiniLM
from sciminilm.training.tokenizer import ScientificTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="SciMiniLM generation demo.")
    parser.add_argument("--prompt", type=str, default="Quantum entanglement arises when")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--gamma", type=int, default=4)
    args = parser.parse_args()

    tok = ScientificTokenizer()
    cfg = ModelConfig.tiny_test(vocab_size=tok.vocab_size, max_seq_len=256)
    target = SciMiniLM(cfg)

    ids = tok.encode(args.prompt, add_bos=True)
    prompt_tensor = torch.tensor([ids], dtype=torch.long)

    if args.speculative:
        draft_cfg = ModelConfig.tiny_test(
            vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=64, max_seq_len=256
        )
        draft = SciMiniLM(draft_cfg)
        decoder = SpeculativeDecoder(target, draft)
        res = decoder.generate(prompt_tensor, max_new_tokens=args.max_new_tokens, gamma=args.gamma)
        print(
            f"Speculative: generated={res.num_generated_tokens}, "
            f"acceptance_rate={res.acceptance_rate:.2%}, "
            f"target_calls={res.num_target_forward_calls}"
        )
    else:
        engine = InferenceEngine(target)
        res = engine.generate(prompt_tensor, max_new_tokens=args.max_new_tokens, top_k=20, top_p=0.9)
        print(
            f"Autoregressive: generated={res.num_generated_tokens}, "
            f"tok/s={res.tokens_per_second:.1f}, kv_cache={res.kv_cache_bytes} bytes"
        )


if __name__ == "__main__":
    main()
