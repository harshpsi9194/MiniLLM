"""Memory footprint measurement and perplexity evaluation harness for quantization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from sciminilm.config import QuantConfig
from sciminilm.quant.linear import (
    QuantizedEmbedding,
    QuantizedLinear,
    quantize_model,
)


@dataclass
class MemoryReport:
    """Memory breakdown and compression ratios for a model."""

    total_bytes: int
    param_bytes: int
    buffer_bytes: int
    linear_bytes: int
    fp32_baseline_bytes: int
    fp16_baseline_bytes: int
    reduction_vs_fp32: float
    reduction_vs_fp16: float
    linear_fp16_baseline_bytes: int
    linear_reduction_vs_fp16: float

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024.0 * 1024.0)


@dataclass
class QuantEvalResult:
    """Evaluation summary comparing a quantized model against a float baseline."""

    name: str
    bits: Optional[int]
    perplexity: float
    ppl_degradation_pct: float
    memory: MemoryReport
    meets_memory_target_vs_fp32: bool
    meets_linear_memory_target_vs_fp16: bool
    meets_ppl_target: bool


def _logical_param_count(model: nn.Module) -> tuple[int, int]:
    """Return (total_logical_params, non_tied_linear_logical_params) deduplicating tied weights."""
    seen_ptrs = set()
    total_params = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            total_params += p.numel()

    linear_params = 0
    seen_linear_ptrs = set()
    emb_ptrs = {
        m.weight.data_ptr()
        for m in model.modules()
        if isinstance(m, nn.Embedding)
    }

    for m in model.modules():
        if isinstance(m, nn.Linear):
            ptr = m.weight.data_ptr()
            if ptr not in emb_ptrs and ptr not in seen_linear_ptrs:
                seen_linear_ptrs.add(ptr)
                linear_params += m.in_features * m.out_features
            if m.bias is not None:
                linear_params += m.out_features
        elif isinstance(m, QuantizedLinear):
            linear_params += m.in_features * m.out_features
            if m.bias is not None:
                linear_params += m.out_features
            if m.qweight.numel() > 0:
                total_params += m.in_features * m.out_features
        elif isinstance(m, QuantizedEmbedding):
            if m.qweight.numel() > 0:
                total_params += m.num_embeddings * m.embedding_dim

    return total_params, linear_params


def measure_model_memory(
    model: nn.Module,
    reference_fp32_bytes: Optional[int] = None,
    reference_fp16_bytes: Optional[int] = None,
    reference_linear_fp16_bytes: Optional[int] = None,
) -> MemoryReport:
    """Measure exact parameter + buffer footprint (deduplicating tied parameters)."""
    seen_ptrs = set()
    param_bytes = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            param_bytes += p.numel() * p.element_size()

    buffer_bytes = 0
    for mod in model.modules():
        non_persistent = getattr(mod, "_non_persistent_buffers_set", set())
        for buf_name, b in mod._buffers.items():
            if b is None or buf_name in non_persistent:
                continue
            ptr = b.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                buffer_bytes += b.numel() * b.element_size()

    emb_ptrs = {
        m.weight.data_ptr()
        for m in model.modules()
        if isinstance(m, nn.Embedding)
    }
    seen_linear_ptrs = set()
    linear_bytes = 0
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            linear_bytes += m.memory_bytes()
        elif isinstance(m, nn.Linear):
            ptr = m.weight.data_ptr()
            if ptr not in emb_ptrs and ptr not in seen_linear_ptrs:
                seen_linear_ptrs.add(ptr)
                linear_bytes += m.weight.numel() * m.weight.element_size()
            if m.bias is not None:
                linear_bytes += m.bias.numel() * m.bias.element_size()

    total_bytes = param_bytes + buffer_bytes

    if (
        reference_fp32_bytes is None
        or reference_fp16_bytes is None
        or reference_linear_fp16_bytes is None
    ):
        total_logical, linear_logical = _logical_param_count(model)
        if reference_fp32_bytes is None:
            reference_fp32_bytes = total_logical * 4
        if reference_fp16_bytes is None:
            reference_fp16_bytes = total_logical * 2
        if reference_linear_fp16_bytes is None:
            reference_linear_fp16_bytes = linear_logical * 2

    reduction_vs_fp32 = reference_fp32_bytes / max(1, total_bytes)
    reduction_vs_fp16 = reference_fp16_bytes / max(1, total_bytes)
    linear_reduction_vs_fp16 = reference_linear_fp16_bytes / max(1, linear_bytes)

    return MemoryReport(
        total_bytes=total_bytes,
        param_bytes=param_bytes,
        buffer_bytes=buffer_bytes,
        linear_bytes=linear_bytes,
        fp32_baseline_bytes=reference_fp32_bytes,
        fp16_baseline_bytes=reference_fp16_bytes,
        reduction_vs_fp32=reduction_vs_fp32,
        reduction_vs_fp16=reduction_vs_fp16,
        linear_fp16_baseline_bytes=reference_linear_fp16_bytes,
        linear_reduction_vs_fp16=linear_reduction_vs_fp16,
    )


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    batches: Iterable[torch.Tensor],
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> float:
    """Compute token-weighted perplexity across input_ids batches."""
    was_training = model.training
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = (
                next(model.buffers()).device
                if any(True for _ in model.buffers())
                else torch.device("cpu")
            )

    total_nll = 0.0
    total_tokens = 0

    for step, batch in enumerate(batches):
        if max_batches is not None and step >= max_batches:
            break
        if isinstance(batch, (tuple, list)):
            input_ids = batch[0]
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
        else:
            input_ids = batch

        input_ids = input_ids.to(device)
        if input_ids.shape[1] < 2:
            continue
        out = model(input_ids, labels=input_ids)
        assert out.loss is not None
        n_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
        total_nll += float(out.loss.item()) * n_tokens
        total_tokens += n_tokens

    if was_training:
        model.train()

    if total_tokens == 0:
        raise ValueError("No valid tokens (seq_len >= 2) provided for perplexity evaluation")

    avg_nll = total_nll / total_tokens
    return math.exp(min(avg_nll, 80.0))


def evaluate_quantization(
    model: nn.Module,
    eval_batches: Sequence[torch.Tensor],
    configs: Optional[Dict[str, QuantConfig]] = None,
    device: Optional[torch.device] = None,
    target_memory_reduction: float = 3.1,
    max_ppl_degradation_pct: float = 6.0,
) -> List[QuantEvalResult]:
    """Evaluate baseline vs quantized variants on memory footprint and perplexity."""
    if device is not None:
        model = model.to(device)

    base_mem = measure_model_memory(model)
    base_ppl = evaluate_perplexity(model, eval_batches, device=device)

    results: List[QuantEvalResult] = [
        QuantEvalResult(
            name="baseline",
            bits=None,
            perplexity=base_ppl,
            ppl_degradation_pct=0.0,
            memory=base_mem,
            meets_memory_target_vs_fp32=base_mem.reduction_vs_fp32 >= target_memory_reduction,
            meets_linear_memory_target_vs_fp16=(
                base_mem.linear_reduction_vs_fp16 >= target_memory_reduction
            ),
            meets_ppl_target=True,
        )
    ]

    if configs is None:
        configs = {
            "int8_per_channel": QuantConfig(bits=8, symmetric=True),
            "int4_g64": QuantConfig(
                bits=4, group_size=64, symmetric=True, quantize_embeddings=True
            ),
        }

    for name, qcfg in configs.items():
        q_model = quantize_model(model, quant_config=qcfg, inplace=False)
        if device is not None:
            q_model = q_model.to(device)
        q_mem = measure_model_memory(
            q_model,
            reference_fp32_bytes=base_mem.fp32_baseline_bytes,
            reference_fp16_bytes=base_mem.fp16_baseline_bytes,
            reference_linear_fp16_bytes=base_mem.linear_fp16_baseline_bytes,
        )
        q_ppl = evaluate_perplexity(q_model, eval_batches, device=device)
        degradation_pct = ((q_ppl - base_ppl) / max(base_ppl, 1e-12)) * 100.0
        results.append(
            QuantEvalResult(
                name=name,
                bits=qcfg.bits,
                perplexity=q_ppl,
                ppl_degradation_pct=degradation_pct,
                memory=q_mem,
                meets_memory_target_vs_fp32=q_mem.reduction_vs_fp32 >= target_memory_reduction,
                meets_linear_memory_target_vs_fp16=(
                    q_mem.linear_reduction_vs_fp16 >= target_memory_reduction
                ),
                meets_ppl_target=degradation_pct <= max_ppl_degradation_pct,
            )
        )

    return results
