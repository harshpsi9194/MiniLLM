"""Tests for INT8/INT4 weight quantization, memory footprint, and perplexity harness."""

import pytest
import torch
import torch.nn as nn

from sciminilm.config import ModelConfig, QuantConfig
from sciminilm.model import SciMiniLM
from sciminilm.quant.evaluate import evaluate_perplexity, evaluate_quantization, measure_model_memory
from sciminilm.quant.linear import QuantizedLinear, quantize_model
from sciminilm.quant.quantizer import (
    dequantize_int4_grouped,
    dequantize_int8_per_channel,
    quantize_int4_grouped,
    quantize_int8_per_channel,
)


@pytest.mark.parametrize("symmetric", [True, False])
def test_int8_per_channel_roundtrip(symmetric: bool):
    torch.manual_seed(0)
    w = torch.randn(32, 64) * 0.5
    qt = quantize_int8_per_channel(w, symmetric=symmetric)
    assert qt.qdata.dtype == torch.int8
    w_hat = dequantize_int8_per_channel(qt)
    rel_err = (w - w_hat).norm() / w.norm()
    assert rel_err < 0.02


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("in_features", [64, 70])
def test_int4_grouped_packing_and_roundtrip(symmetric: bool, in_features: int):
    torch.manual_seed(1)
    w = torch.randn(16, in_features) * 0.25
    qt = quantize_int4_grouped(w, group_size=32, symmetric=symmetric)
    assert qt.qdata.dtype == torch.uint8
    assert qt.qdata.shape == (16, qt.padded_in_features // 2)

    w_hat = dequantize_int4_grouped(qt)
    assert w_hat.shape == (16, in_features)
    rel_err = (w - w_hat).norm() / w.norm()
    assert rel_err < 0.15


def test_quantized_linear_output_closeness_and_memory():
    torch.manual_seed(2)
    lin = nn.Linear(128, 64, bias=True)
    qlin_8 = QuantizedLinear.from_linear(lin, bits=8)
    qlin_4 = QuantizedLinear.from_linear(lin, bits=4, group_size=64)

    x = torch.randn(4, 10, 128)
    y_ref = lin(x)
    y_8 = qlin_8(x)
    y_4 = qlin_4(x)

    assert (y_ref - y_8).norm() / y_ref.norm() < 0.02
    assert (y_ref - y_4).norm() / y_ref.norm() < 0.15

    fp32_bytes = lin.weight.numel() * 4 + lin.bias.numel() * 4
    fp16_bytes = lin.weight.numel() * 2 + lin.bias.numel() * 2
    assert qlin_8.memory_bytes() < fp32_bytes / 3.5
    assert qlin_4.memory_bytes() < fp16_bytes / 3.2


def test_quantize_model_memory_reduction_and_perplexity_harness():
    torch.manual_seed(3)
    cfg = ModelConfig.tiny_test(d_model=128, n_layers=3, n_heads=4, n_kv_heads=2, d_ff=256)
    model = SciMiniLM(cfg)

    # Train briefly on a structured repeating pattern so PPL is well-calibrated (< random)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    pattern = torch.arange(0, 64, dtype=torch.long).unsqueeze(0).repeat(4, 1) % cfg.vocab_size
    model.train()
    for _ in range(30):
        opt.zero_grad()
        out = model(pattern, labels=pattern)
        assert out.loss is not None
        out.loss.backward()
        opt.step()

    eval_batches = [pattern[:2]]
    results = evaluate_quantization(
        model,
        eval_batches,
        configs={
            "int8": QuantConfig(bits=8, symmetric=True),
            "int4_g64": QuantConfig(bits=4, group_size=64, symmetric=True),
        },
    )
    by_name = {r.name: r for r in results}

    assert by_name["int8"].memory.reduction_vs_fp32 > 2.5
    assert by_name["int4_g64"].memory.linear_reduction_vs_fp16 >= 3.1
    assert by_name["int8"].ppl_degradation_pct < 6.0
    assert by_name["int4_g64"].ppl_degradation_pct < 6.0
    assert by_name["int4_g64"].meets_ppl_target
    assert by_name["int4_g64"].meets_linear_memory_target_vs_fp16

    # Also verify full-model (including tied embeddings) INT4 quantization achieves >3.1x vs FP16
    res_full = evaluate_quantization(
        model,
        eval_batches,
        configs={
            "int4_full": QuantConfig(
                bits=4, group_size=64, symmetric=True, quantize_embeddings=True
            )
        },
    )
    full_entry = {r.name: r for r in res_full}["int4_full"]
    assert full_entry.memory.reduction_vs_fp16 >= 3.1
    assert full_entry.memory.reduction_vs_fp32 >= 6.0
    assert full_entry.ppl_degradation_pct < 6.0


def test_quantize_model_skips_lm_head_and_preserves_original():
    cfg = ModelConfig.tiny_test()
    model = SciMiniLM(cfg)
    base_mem = measure_model_memory(model)
    q_model = quantize_model(model, QuantConfig(bits=4, group_size=32), inplace=False)

    # Original model untouched
    assert isinstance(model.layers[0].attn.q_proj, nn.Linear)
    # Quantized model replaced attention/MLP linears but skipped lm_head
    assert isinstance(q_model.layers[0].attn.q_proj, QuantizedLinear)
    assert isinstance(q_model.lm_head, nn.Linear)
    assert measure_model_memory(q_model).total_bytes < base_mem.total_bytes
