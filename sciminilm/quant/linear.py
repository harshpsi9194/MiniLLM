"""Quantized Linear and Embedding layers plus model quantization helper."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sciminilm.config import QuantConfig
from sciminilm.quant.quantizer import (
    QuantizedTensor,
    dequantize_int4_grouped,
    dequantize_int8_per_channel,
    quantize_weight,
)


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear storing INT8 or packed INT4 weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 4,
        group_size: int = 64,
        symmetric: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.register_buffer("qweight", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("scales", torch.empty(0, dtype=torch.float16))
        self.register_buffer("zero_points", None)
        self.padded_in_features: int = in_features

    def load_from_float_weight(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        qt = quantize_weight(
            weight,
            bits=self.bits,
            group_size=self.group_size,
            symmetric=self.symmetric,
        )
        self.qweight = qt.qdata.to(weight.device)
        self.scales = qt.scales.to(weight.device)
        self.zero_points = (
            qt.zero_points.to(weight.device) if qt.zero_points is not None else None
        )
        self.padded_in_features = qt.padded_in_features
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 64,
        symmetric: bool = True,
    ) -> "QuantizedLinear":
        """Create a QuantizedLinear layer from a floating-point nn.Linear."""
        qlin = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
        )
        qlin.load_from_float_weight(linear.weight, linear.bias)
        return qlin

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        qt = QuantizedTensor(
            qdata=self.qweight,
            scales=self.scales,
            zero_points=self.zero_points,
            bits=self.bits,
            group_size=self.group_size if self.bits == 4 else self.in_features,
            symmetric=self.symmetric,
            orig_shape=(self.out_features, self.in_features),
            padded_in_features=self.padded_in_features,
        )
        if self.bits == 8:
            return dequantize_int8_per_channel(qt, dtype=dtype)
        return dequantize_int4_grouped(qt, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def memory_bytes(self) -> int:
        total = self.qweight.numel() * self.qweight.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.zero_points is not None:
            total += self.zero_points.numel() * self.zero_points.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, symmetric={self.symmetric}"
        )


class QuantizedEmbedding(nn.Module):
    """Quantized token embedding storing INT8 or packed INT4 rows."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bits: int = 4,
        group_size: int = 64,
        symmetric: bool = True,
        out_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.out_dtype = out_dtype

        self.register_buffer("qweight", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("scales", torch.empty(0, dtype=torch.float16))
        self.register_buffer("zero_points", None)
        self.padded_in_features: int = embedding_dim

    @classmethod
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        bits: int = 4,
        group_size: int = 64,
        symmetric: bool = True,
    ) -> "QuantizedEmbedding":
        qemb = cls(
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            out_dtype=embedding.weight.dtype,
        )
        qt = quantize_weight(
            embedding.weight,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
        )
        qemb.qweight = qt.qdata.to(embedding.weight.device)
        qemb.scales = qt.scales.to(embedding.weight.device)
        qemb.zero_points = (
            qt.zero_points.to(embedding.weight.device) if qt.zero_points is not None else None
        )
        qemb.padded_in_features = qt.padded_in_features
        return qemb

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        qt = QuantizedTensor(
            qdata=self.qweight,
            scales=self.scales,
            zero_points=self.zero_points,
            bits=self.bits,
            group_size=self.group_size if self.bits == 4 else self.embedding_dim,
            symmetric=self.symmetric,
            orig_shape=(self.num_embeddings, self.embedding_dim),
            padded_in_features=self.padded_in_features,
        )
        if self.bits == 8:
            return dequantize_int8_per_channel(qt, dtype=dtype)
        return dequantize_int4_grouped(qt, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize(dtype=self.out_dtype)
        return F.embedding(input_ids, weight)

    def memory_bytes(self) -> int:
        total = self.qweight.numel() * self.qweight.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.zero_points is not None:
            total += self.zero_points.numel() * self.zero_points.element_size()
        return total


class TiedQuantizedLMHead(nn.Module):
    """LM head sharing quantized weights with a QuantizedEmbedding."""

    def __init__(self, embedding: QuantizedEmbedding) -> None:
        super().__init__()
        self.embedding = embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.embedding.dequantize(dtype=x.dtype)
        return F.linear(x, weight)


def _should_skip(name: str, skip_modules: tuple[str, ...]) -> bool:
    return any(name == s or name.endswith(f".{s}") or s in name.split(".") for s in skip_modules)


def quantize_model(
    model: nn.Module,
    quant_config: Optional[QuantConfig] = None,
    inplace: bool = False,
) -> nn.Module:
    """Replace eligible nn.Linear (and optionally nn.Embedding) layers with quantized layers."""
    if quant_config is None:
        quant_config = QuantConfig()

    target_model = model if inplace else copy.deepcopy(model)

    tied_emb_lm = (
        hasattr(target_model, "tok_embeddings")
        and hasattr(target_model, "lm_head")
        and isinstance(target_model.tok_embeddings, nn.Embedding)
        and isinstance(target_model.lm_head, nn.Linear)
        and target_model.tok_embeddings.weight.data_ptr()
        == target_model.lm_head.weight.data_ptr()
    )

    if quant_config.quantize_embeddings and hasattr(target_model, "tok_embeddings"):
        emb = target_model.tok_embeddings
        if isinstance(emb, nn.Embedding):
            q_emb = QuantizedEmbedding.from_embedding(
                emb,
                bits=quant_config.bits,
                group_size=quant_config.group_size,
                symmetric=quant_config.symmetric,
            )
            target_model.tok_embeddings = q_emb
            if tied_emb_lm:
                target_model.lm_head = TiedQuantizedLMHead(q_emb)

    def _replace(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                if _should_skip(full_name, quant_config.skip_modules):
                    continue
                q_layer = QuantizedLinear.from_linear(
                    child,
                    bits=quant_config.bits,
                    group_size=quant_config.group_size,
                    symmetric=quant_config.symmetric,
                )
                setattr(module, child_name, q_layer)
            else:
                _replace(child, prefix=full_name)

    _replace(target_model)
    return target_model
