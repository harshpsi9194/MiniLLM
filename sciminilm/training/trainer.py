"""Complete training loop for SciMiniLM (500M-token schedule, cosine LR, AMP, checkpointing)."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sciminilm.config import TrainConfig
from sciminilm.model import SciMiniLM
from sciminilm.quant.evaluate import evaluate_perplexity


def cosine_lr_with_warmup(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Compute learning rate with linear warmup followed by cosine decay."""
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))
    if step >= max_steps:
        return min_lr
    decay_steps = max(1, max_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


def build_optimizer(
    model: nn.Module,
    learning_rate: float = 6e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """Build AdamW optimizer separating decayed matrix weights from non-decayed 1D params."""
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    seen = set()

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        if p.ndim >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas)


@dataclass
class StepLog:
    """Metrics recorded for a single optimizer step."""

    step: int
    loss: float
    lr: float
    tokens_seen: int
    step_time_s: float
    val_perplexity: Optional[float] = None


@dataclass
class TrainSummary:
    """Summary returned at the end of a training run."""

    steps_completed: int
    tokens_seen: int
    final_loss: float
    final_val_perplexity: Optional[float]
    history: List[StepLog] = field(default_factory=list)


class Trainer:
    """End-to-end pretraining engine for SciMiniLM."""

    def __init__(
        self,
        model: SciMiniLM,
        config: TrainConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

        self.optimizer = build_optimizer(
            self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        self.use_cuda = self.device.type == "cuda"
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.amp_dtype = dtype_map[config.dtype]
        self.use_amp = self.use_cuda and self.amp_dtype in (torch.float16, torch.bfloat16)
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=(self.use_cuda and self.amp_dtype == torch.float16))
        )

        self.global_step = 0
        self.tokens_seen = 0

    def _autocast_ctx(self):
        if self.use_amp:
            return torch.amp.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()

    def _infinite_batches(self) -> Iterator[torch.Tensor]:
        while True:
            yielded = False
            for batch in self.train_loader:
                yielded = True
                yield batch
            if not yielded:
                raise RuntimeError("train_loader yielded zero batches")

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        """Save model, optimizer, scaler, and training progress checkpoint."""
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "model_config": self.model.config.to_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    def load_checkpoint(self, path: Union[str, Path]) -> Dict:
        """Restore model, optimizer, and step counters from checkpoint."""
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scaler_state_dict" in payload:
            self.scaler.load_state_dict(payload["scaler_state_dict"])
        self.global_step = int(payload.get("global_step", 0))
        self.tokens_seen = int(payload.get("tokens_seen", 0))
        return payload

    def train(self, max_steps: Optional[int] = None) -> TrainSummary:
        """Run the training loop for `max_steps` (or `config.max_steps` for 500M tokens)."""
        torch.manual_seed(self.config.seed)
        target_steps = max_steps if max_steps is not None else self.config.max_steps

        batch_iter = self._infinite_batches()
        history: List[StepLog] = []
        last_loss = 0.0
        last_val_ppl: Optional[float] = None

        self.model.train()
        while self.global_step < target_steps:
            t0 = time.perf_counter()
            lr = cosine_lr_with_warmup(
                step=self.global_step,
                warmup_steps=self.config.warmup_steps,
                max_steps=target_steps,
                max_lr=self.config.learning_rate,
                min_lr=self.config.min_learning_rate,
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            step_tokens = 0

            for _ in range(self.config.grad_accum_steps):
                batch = next(batch_iter)
                if isinstance(batch, (tuple, list)):
                    input_ids = batch[0]
                elif isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                else:
                    input_ids = batch
                input_ids = input_ids.to(self.device, non_blocking=self.use_cuda)

                with self._autocast_ctx():
                    out = self.model(input_ids, labels=input_ids)
                    assert out.loss is not None
                    loss = out.loss / float(self.config.grad_accum_steps)

                self.scaler.scale(loss).backward()
                accum_loss += float(loss.item())
                step_tokens += int(input_ids.numel())

            if self.config.grad_clip > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1
            self.tokens_seen += step_tokens
            last_loss = accum_loss

            val_ppl: Optional[float] = None
            if (
                self.val_loader is not None
                and self.config.eval_interval > 0
                and self.global_step % self.config.eval_interval == 0
            ):
                val_ppl = evaluate_perplexity(
                    self.model,
                    self.val_loader,
                    device=self.device,
                    max_batches=self.config.eval_steps,
                )
                last_val_ppl = val_ppl

            if (
                self.config.save_interval > 0
                and self.global_step % self.config.save_interval == 0
            ):
                ckpt_file = Path(self.config.checkpoint_dir) / f"step_{self.global_step:07d}.pt"
                self.save_checkpoint(ckpt_file)

            history.append(
                StepLog(
                    step=self.global_step,
                    loss=last_loss,
                    lr=lr,
                    tokens_seen=self.tokens_seen,
                    step_time_s=time.perf_counter() - t0,
                    val_perplexity=val_ppl,
                )
            )

        return TrainSummary(
            steps_completed=self.global_step,
            tokens_seen=self.tokens_seen,
            final_loss=last_loss,
            final_val_perplexity=last_val_ppl,
            history=history,
        )
