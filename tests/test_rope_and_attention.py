"""Tests for RotaryEmbedding and MHA/GQA/MQA CausalSelfAttention with FlashAttention/fallback."""

import pytest
import torch

from sciminilm.config import ModelConfig
from sciminilm.inference.kv_cache import KVCache
from sciminilm.layers.attention import CausalSelfAttention
from sciminilm.layers.rope import RotaryEmbedding


def test_rope_norm_preservation_and_offset_consistency():
    torch.manual_seed(0)
    rope = RotaryEmbedding(dim=16, max_seq_len=64)
    q = torch.randn(2, 4, 12, 16)
    k = torch.randn(2, 2, 12, 16)

    q_rot, k_rot = rope(q, k, position_offset=0)
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-5)

    # Offset=5 slice must match position 5..12 of full sequence
    q_sub, k_sub = rope(q[:, :, 5:, :], k[:, :, 5:, :], position_offset=5)
    assert torch.allclose(q_sub, q_rot[:, :, 5:, :], atol=1e-6)
    assert torch.allclose(k_sub, k_rot[:, :, 5:, :], atol=1e-6)


@pytest.mark.parametrize("variant", ["mha", "gqa", "mqa"])
def test_attention_sdpa_matches_manual_fallback(variant: str):
    torch.manual_seed(1)
    cfg = ModelConfig.tiny_test(attention_variant=variant, use_flash_attention=True)
    attn_flash = CausalSelfAttention(cfg, layer_idx=0)
    attn_flash.eval()

    cfg_manual = ModelConfig(**{**cfg.to_dict(), "use_flash_attention": False})
    attn_manual = CausalSelfAttention(cfg_manual, layer_idx=0)
    attn_manual.load_state_dict(attn_flash.state_dict())
    attn_manual.eval()

    x = torch.randn(2, 15, cfg.d_model)
    out_flash = attn_flash(x)
    out_manual = attn_manual(x)
    assert torch.allclose(out_flash, out_manual, atol=1e-5)


def test_causal_masking_strictly_prevents_future_leakage():
    torch.manual_seed(2)
    cfg = ModelConfig.tiny_test(attention_variant="gqa")
    attn = CausalSelfAttention(cfg, layer_idx=0)
    attn.eval()

    x1 = torch.randn(1, 10, cfg.d_model)
    x2 = x1.clone()
    x2[:, 6:, :] = torch.randn(1, 4, cfg.d_model)

    y1 = attn(x1)
    y2 = attn(x2)
    assert torch.allclose(y1[:, :6, :], y2[:, :6, :], atol=1e-6)
    assert not torch.allclose(y1[:, 6:, :], y2[:, 6:, :], atol=1e-3)


def test_attention_chunked_prefill_with_kv_cache():
    torch.manual_seed(3)
    cfg = ModelConfig.tiny_test(attention_variant="gqa")
    attn = CausalSelfAttention(cfg, layer_idx=0)
    attn.eval()

    x = torch.randn(1, 10, cfg.d_model)
    full_out = attn(x)

    cache = KVCache(n_layers=1)
    p1 = attn(x[:, :6, :], kv_cache=cache)
    p2 = attn(x[:, 6:, :], kv_cache=cache)
    cat_out = torch.cat([p1, p2], dim=1)
    assert torch.allclose(cat_out, full_out, atol=1e-5)
