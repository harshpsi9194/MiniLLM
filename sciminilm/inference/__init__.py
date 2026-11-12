"""Autoregressive inference engine, KV cache, sampling, and speculative decoding."""

from sciminilm.inference.engine import GenerationResult, InferenceEngine
from sciminilm.inference.kv_cache import KVCache
from sciminilm.inference.sampling import (
    apply_top_k,
    apply_top_p,
    compute_sampling_probs,
    sample_next_token,
)
from sciminilm.inference.speculative import SpeculativeDecoder, SpeculativeResult

__all__ = [
    "GenerationResult",
    "InferenceEngine",
    "KVCache",
    "SpeculativeDecoder",
    "SpeculativeResult",
    "apply_top_k",
    "apply_top_p",
    "compute_sampling_probs",
    "sample_next_token",
]
