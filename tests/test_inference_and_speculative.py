"""Tests for KVCache, Top-K/Top-P sampling, InferenceEngine, and SpeculativeDecoder."""

import pytest
import torch

from sciminilm.config import ModelConfig
from sciminilm.inference.engine import InferenceEngine
from sciminilm.inference.kv_cache import KVCache
from sciminilm.inference.sampling import apply_top_k, apply_top_p, compute_sampling_probs
from sciminilm.inference.speculative import SpeculativeDecoder
from sciminilm.model import SciMiniLM


def test_kv_cache_update_and_truncate():
    cache = KVCache(n_layers=2)
    k1 = torch.randn(1, 2, 5, 16)
    v1 = torch.randn(1, 2, 5, 16)
    k_full, v_full = cache.update(0, k1, v1)
    assert k_full.shape == (1, 2, 5, 16)
    assert cache.seq_len(0) == 5
    assert cache.seq_len(1) == 0

    k2 = torch.randn(1, 2, 3, 16)
    v2 = torch.randn(1, 2, 3, 16)
    k_full, _ = cache.update(0, k2, v2)
    assert k_full.shape == (1, 2, 8, 16)
    assert cache.memory_bytes() > 0

    cache.truncate(4)
    assert cache.seq_len(0) == 4
    cache.clear()
    assert cache.seq_len(0) == 0
    assert cache.memory_bytes() == 0


def test_top_k_and_top_p_filtering():
    logits = torch.tensor([[10.0, 5.0, 1.0, -5.0, -10.0]])
    tk = apply_top_k(logits, top_k=2)
    assert torch.isfinite(tk[0, :2]).all()
    assert torch.isneginf(tk[0, 2:]).all()

    probs = compute_sampling_probs(logits, temperature=1.0, top_k=0, top_p=0.9)
    assert probs[0, 0] > 0.9
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    # Tail tokens should be filtered out by nucleus threshold
    tp = apply_top_p(logits, top_p=0.5)
    assert torch.isfinite(tp[0, 0])
    assert torch.isneginf(tp[0, 1:]).all()


def test_kv_cached_decoding_matches_un_cached_greedy():
    torch.manual_seed(10)
    cfg = ModelConfig.tiny_test(attention_variant="gqa", max_seq_len=64)
    model = SciMiniLM(cfg)
    model.eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    engine = InferenceEngine(model)
    res = engine.generate(prompt, max_new_tokens=10, do_sample=False)
    assert res.generated_tokens.shape == (1, 10)
    assert res.sequences.shape == (1, 18)

    # Compare step-by-step against full un-cached forward pass
    seq = prompt.clone()
    with torch.no_grad():
        for _ in range(10):
            logits = model(seq).logits[:, -1, :]
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
    assert torch.equal(res.sequences, seq)


def test_inference_engine_eos_and_sampling():
    torch.manual_seed(11)
    cfg = ModelConfig.tiny_test(max_seq_len=32)
    model = SciMiniLM(cfg)
    engine = InferenceEngine(model)

    prompt = torch.randint(0, cfg.vocab_size, (2, 4))
    gen = torch.Generator().manual_seed(123)
    res = engine.generate(
        prompt,
        max_new_tokens=8,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        do_sample=True,
        generator=gen,
    )
    assert res.sequences.shape == (2, 12)
    assert res.kv_cache_bytes > 0

    # Verify early stopping when eos_token_id is emitted on the first step
    first_greedy = int(
        engine.generate(prompt[:1], max_new_tokens=1, do_sample=False).generated_tokens[0, 0].item()
    )
    eos_res = engine.generate(
        prompt[:1],
        max_new_tokens=10,
        do_sample=False,
        eos_token_id=first_greedy,
    )
    assert eos_res.num_generated_tokens == 1
    assert int(eos_res.generated_tokens[0, 0].item()) == first_greedy


def test_speculative_greedy_matches_target_exactly():
    torch.manual_seed(20)
    target_cfg = ModelConfig.tiny_test(attention_variant="gqa", max_seq_len=64)
    draft_cfg = ModelConfig.tiny_test(
        d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=64, max_seq_len=64
    )
    target = SciMiniLM(target_cfg)
    draft = SciMiniLM(draft_cfg)

    prompt = torch.randint(0, target_cfg.vocab_size, (1, 6))
    baseline = InferenceEngine(target).generate(prompt, max_new_tokens=15, do_sample=False)

    spec = SpeculativeDecoder(target, draft)
    spec_res = spec.generate(prompt, max_new_tokens=15, gamma=4, do_sample=False)

    assert torch.equal(spec_res.sequences, baseline.sequences)
    assert spec_res.num_generated_tokens == 15


def test_speculative_self_draft_has_100pct_acceptance_and_stochastic_works():
    torch.manual_seed(21)
    cfg = ModelConfig.tiny_test(max_seq_len=64)
    model = SciMiniLM(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))

    # Identical draft & target in greedy mode -> 100% acceptance rate
    spec_same = SpeculativeDecoder(model, model)
    res_greedy = spec_same.generate(prompt, max_new_tokens=12, gamma=4, do_sample=False)
    assert res_greedy.acceptance_rate == pytest.approx(1.0)
    assert res_greedy.num_generated_tokens == 12

    # Stochastic mode with top_k / top_p
    gen = torch.Generator().manual_seed(99)
    res_sample = spec_same.generate(
        prompt,
        max_new_tokens=12,
        gamma=3,
        temperature=0.8,
        top_k=16,
        top_p=0.9,
        do_sample=True,
        generator=gen,
    )
    assert res_sample.num_generated_tokens == 12
    assert res_sample.sequences.shape == (1, 17)
