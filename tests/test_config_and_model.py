"""Tests for ModelConfig presets (77M & tiny), TrainConfig, and SciMiniLM architecture."""

import pytest
import torch

from sciminilm.config import ModelConfig, TrainConfig
from sciminilm.model import SciMiniLM


def test_77m_preset_parameter_count():
    cfg = ModelConfig.sciminilm_77m(attention_variant="mha")
    assert cfg.attention_variant == "mha"
    assert cfg.head_dim == 64
    estimated = cfg.estimate_num_params()
    assert 76_000_000 <= estimated <= 78_000_000

    # Instantiate on meta device so we verify exact parameter count without huge RAM alloc
    with torch.device("meta"):
        model = SciMiniLM(cfg)
    actual = model.num_parameters()
    assert actual == estimated == 77_020_800


def test_attention_variant_presets():
    gqa = ModelConfig.sciminilm_77m(attention_variant="gqa")
    mqa = ModelConfig.sciminilm_77m(attention_variant="mqa")
    assert gqa.attention_variant == "gqa"
    assert gqa.n_kv_heads == 2
    assert mqa.attention_variant == "mqa"
    assert mqa.n_kv_heads == 1
    assert gqa.estimate_num_params() < ModelConfig.sciminilm_77m("mha").estimate_num_params()
    assert mqa.estimate_num_params() < gqa.estimate_num_params()


def test_invalid_model_configs_raise():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=65, n_heads=4, n_kv_heads=2)
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=64, n_heads=4, n_kv_heads=3)
    with pytest.raises(ValueError):
        ModelConfig(d_model=64, n_heads=4, n_kv_heads=8)


def test_tiny_model_forward_loss_and_backward():
    cfg = ModelConfig.tiny_test(attention_variant="gqa")
    model = SciMiniLM(cfg)
    assert model.num_parameters() == cfg.estimate_num_params()
    assert model.tok_embeddings.weight is model.lm_head.weight

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16), dtype=torch.long)
    out = model(input_ids, labels=input_ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.loss is not None
    assert torch.isfinite(out.loss)

    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_tiny_model_cuda_end_to_end():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = ModelConfig.tiny_test(attention_variant="gqa", use_triton_rmsnorm=True, use_flash_attention=True)
    model = SciMiniLM(cfg).cuda()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16), dtype=torch.long, device="cuda")
    out = model(input_ids, labels=input_ids)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()


def test_train_config_500m_budget():
    tcfg = TrainConfig()
    assert tcfg.total_tokens == 500_000_000
    assert tcfg.tokens_per_step == 8 * 16 * 2048
    assert tcfg.max_steps * tcfg.tokens_per_step >= 500_000_000
