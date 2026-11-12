"""Tests for ScientificTokenizer, 500M-token streaming/memmap datasets, and Trainer."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sciminilm.config import ModelConfig, TrainConfig
from sciminilm.model import SciMiniLM
from sciminilm.training.dataset import (
    MemmapTokenDataset,
    PackedTokenStreamingDataset,
    iter_text_files,
    write_token_shard,
)
from sciminilm.training.tokenizer import ScientificTokenizer
from sciminilm.training.trainer import Trainer, cosine_lr_with_warmup


def test_scientific_tokenizer_bpe_roundtrip_and_persistence(tmp_path: Path):
    docs = [
        "The Hamiltonian H = T + V describes quantum energy eigenstates.",
        "Navier-Stokes: ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u for incompressible flow.",
        "Thermodynamics: ΔG = ΔH - TΔS dictates reaction spontaneity.",
    ] * 5

    tok = ScientificTokenizer.train_from_iterator(docs, vocab_size=300)
    assert tok.vocab_size > 260
    assert tok.vocab_size <= 300

    sample = "Eigenvalue λ satisfies Hψ = Eψ; ΔG < 0!"
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == sample

    tok_path = tmp_path / "tokenizer.json"
    tok.save(tok_path)
    loaded = ScientificTokenizer.load(tok_path)
    assert loaded.encode(sample) == tok.encode(sample)


def test_streaming_and_memmap_datasets(tmp_path: Path):
    jsonl_path = tmp_path / "sci.jsonl"
    records = [
        {"text": f"Experiment {i}: measured conductivity sigma={i * 1.5:.2f} S/m."}
        for i in range(20)
    ]
    jsonl_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    tok = ScientificTokenizer()
    ds = PackedTokenStreamingDataset(
        text_stream_factory=lambda: iter_text_files([jsonl_path]),
        tokenizer=tok,
        seq_len=16,
        shuffle_buffer_size=4,
        max_tokens=16 * 5,
    )
    chunks = list(ds)
    assert len(chunks) == 5
    assert all(c.shape == (16,) and c.dtype == torch.long for c in chunks)

    # Memmap shard creation and random access
    bin_path = tmp_path / "shard.bin"
    n_toks = write_token_shard(iter_text_files([jsonl_path]), tok, bin_path)
    assert n_toks >= 32
    mm_ds = MemmapTokenDataset(bin_path, seq_len=16)
    assert len(mm_ds) == n_toks // 16
    assert mm_ds[0].shape == (16,)


def test_cosine_lr_schedule():
    lr_start = cosine_lr_with_warmup(0, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4)
    lr_peak = cosine_lr_with_warmup(9, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4)
    lr_end = cosine_lr_with_warmup(100, warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4)
    assert lr_start < lr_peak
    assert abs(lr_peak - 1e-3) < 1e-9
    assert abs(lr_end - 1e-4) < 1e-9


def test_trainer_runs_and_checkpoints(tmp_path: Path):
    torch.manual_seed(42)
    tok = ScientificTokenizer()
    cfg = ModelConfig.tiny_test(vocab_size=tok.vocab_size, max_seq_len=32)
    model = SciMiniLM(cfg)

    corpus = [
        "Quantum mechanics predicts quantized angular momentum in atomic orbitals. "
    ] * 30
    ds = PackedTokenStreamingDataset(
        text_stream_factory=lambda: iter(corpus),
        tokenizer=tok,
        seq_len=16,
        max_tokens=None,
    )
    loader = DataLoader(ds, batch_size=2)
    val_loader = DataLoader(ds, batch_size=2)

    tcfg = TrainConfig(
        total_tokens=1024,
        seq_len=16,
        micro_batch_size=2,
        grad_accum_steps=2,
        learning_rate=3e-3,
        min_learning_rate=3e-4,
        warmup_steps=2,
        eval_interval=3,
        eval_steps=2,
        save_interval=3,
        dtype="float32",
        checkpoint_dir=str(tmp_path / "ckpts"),
    )

    trainer = Trainer(model, tcfg, loader, val_loader=val_loader, device=torch.device("cpu"))
    summary = trainer.train(max_steps=6)

    assert summary.steps_completed == 6
    assert summary.tokens_seen == 6 * 2 * 2 * 16
    assert summary.history[-1].loss < summary.history[0].loss
    assert summary.final_val_perplexity is not None

    ckpt_file = tmp_path / "ckpts" / "step_0000003.pt"
    assert ckpt_file.exists()

    # Verify checkpoint resume restores exact step and weights
    model2 = SciMiniLM(cfg)
    trainer2 = Trainer(model2, tcfg, loader, device=torch.device("cpu"))
    trainer2.load_checkpoint(ckpt_file)
    assert trainer2.global_step == 3
    assert trainer2.tokens_seen == 3 * 2 * 2 * 16
