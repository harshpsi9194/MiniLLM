"""Post-training INT8/INT4 weight quantization and evaluation utilities."""

from sciminilm.quant.evaluate import (
    MemoryReport,
    QuantEvalResult,
    evaluate_perplexity,
    evaluate_quantization,
    measure_model_memory,
)
from sciminilm.quant.linear import (
    QuantizedEmbedding,
    QuantizedLinear,
    TiedQuantizedLMHead,
    quantize_model,
)
from sciminilm.quant.quantizer import (
    QuantizedTensor,
    dequantize_int4_grouped,
    dequantize_int8_per_channel,
    dequantize_weight,
    quantize_int4_grouped,
    quantize_int8_per_channel,
    quantize_weight,
)

__all__ = [
    "MemoryReport",
    "QuantEvalResult",
    "QuantizedEmbedding",
    "QuantizedLinear",
    "QuantizedTensor",
    "TiedQuantizedLMHead",
    "dequantize_int4_grouped",
    "dequantize_int8_per_channel",
    "dequantize_weight",
    "evaluate_perplexity",
    "evaluate_quantization",
    "measure_model_memory",
    "quantize_int4_grouped",
    "quantize_int8_per_channel",
    "quantize_model",
    "quantize_weight",
]
