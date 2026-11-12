"""SciMiniLM: 77M-parameter Decoder-Only Scientific Language Model."""

from sciminilm.config import InferenceConfig, ModelConfig, QuantConfig, TrainConfig
from sciminilm.inference import InferenceEngine, KVCache, SpeculativeDecoder
from sciminilm.model import ModelOutput, SciMiniLM, TransformerBlock
from sciminilm.quant import (
    QuantizedLinear,
    evaluate_perplexity,
    evaluate_quantization,
    measure_model_memory,
    quantize_model,
)

__all__ = [
    "InferenceConfig",
    "InferenceEngine",
    "KVCache",
    "ModelConfig",
    "ModelOutput",
    "QuantConfig",
    "QuantizedLinear",
    "SciMiniLM",
    "SpeculativeDecoder",
    "TrainConfig",
    "TransformerBlock",
    "evaluate_perplexity",
    "evaluate_quantization",
    "measure_model_memory",
    "quantize_model",
]
