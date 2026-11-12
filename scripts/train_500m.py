"""Training entrypoint for SciMiniLM-77M on 500M scientific-text tokens."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from sciminilm.config import ModelConfig, TrainConfig
from sciminilm.model import SciMiniLM
from sciminilm.training.dataset import (
    MemmapTokenDataset,
    PackedTokenStreamingDataset,
    iter_text_files,
)
from sciminilm.training.tokenizer import ScientificTokenizer
from sciminilm.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SciMiniLM-77M on 500M scientific tokens.")
    parser.add_argument("--data-files", nargs="+", default=None, help="Paths to .jsonl or .txt files.")
    parser.add_argument("--bin-shard", type=str, default=None, help="Path to pre-tokenized .bin shard.")
    parser.add_argument("--val-bin-shard", type=str, default=None, help="Optional validation .bin shard.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to saved ScientificTokenizer JSON.")
    parser.add_argument("--attention", choices=["mha", "gqa", "mqa"], default="mha")
    parser.add_argument("--total-tokens", type=int, default=500_000_000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/sciminilm_77m")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step cap for short runs.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = ModelConfig.sciminilm_77m(
        attention_variant=args.attention,
        max_seq_len=args.seq_len,
    )
    train_cfg = TrainConfig(
        total_tokens=args.total_tokens,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    model = SciMiniLM(model_cfg)
    print(
        f"SciMiniLM ({model_cfg.attention_variant.upper()}) | "
        f"Params: {model.num_parameters():,} | "
        f"Target tokens: {train_cfg.total_tokens:,} | "
        f"Tokens/step: {train_cfg.tokens_per_step:,} | "
        f"Max steps: {train_cfg.max_steps:,}"
    )

    if args.dry_run:
        return

    tokenizer = (
        ScientificTokenizer.load(args.tokenizer)
        if args.tokenizer is not None
        else ScientificTokenizer()
    )

    if args.bin_shard is not None:
        train_ds = MemmapTokenDataset(args.bin_shard, seq_len=train_cfg.seq_len)
        train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg.micro_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=train_cfg.num_workers,
        )
    elif args.data_files is not None:
        files = [Path(p) for p in args.data_files]
        train_ds = PackedTokenStreamingDataset(
            text_stream_factory=lambda: iter_text_files(files),
            tokenizer=tokenizer,
            seq_len=train_cfg.seq_len,
            max_tokens=train_cfg.total_tokens,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg.micro_batch_size,
            drop_last=True,
            num_workers=train_cfg.num_workers,
        )
    else:
        raise ValueError("Provide either --bin-shard or --data-files (or pass --dry-run).")

    val_loader = None
    if args.val_bin_shard is not None:
        val_ds = MemmapTokenDataset(args.val_bin_shard, seq_len=train_cfg.seq_len)
        val_loader = DataLoader(val_ds, batch_size=train_cfg.micro_batch_size, shuffle=False)

    trainer = Trainer(model, train_cfg, train_loader, val_loader=val_loader)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    summary = trainer.train(max_steps=args.max_steps)
    print(
        f"Training complete: steps={summary.steps_completed}, "
        f"tokens={summary.tokens_seen:,}, final_loss={summary.final_loss:.4f}"
    )


if __name__ == "__main__":
    main()
