"""Speculative decoding (draft + parallel target verification with KV-cache rollback)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from sciminilm.config import InferenceConfig
from sciminilm.inference.kv_cache import KVCache
from sciminilm.inference.sampling import compute_sampling_probs, sample_next_token
from sciminilm.model import SciMiniLM


@dataclass
class SpeculativeResult:
    """Output and diagnostics from speculative decoding."""

    sequences: torch.Tensor
    generated_tokens: torch.Tensor
    num_generated_tokens: int
    total_draft_tokens: int
    accepted_draft_tokens: int
    acceptance_rate: float
    num_target_forward_calls: int
    total_time_s: float
    tokens_per_second: float


class SpeculativeDecoder:
    """Exact speculative decoding using a small draft model and a target SciMiniLM."""

    def __init__(self, target_model: SciMiniLM, draft_model: SciMiniLM) -> None:
        if target_model.config.vocab_size != draft_model.config.vocab_size:
            raise ValueError("Target and draft models must share the same vocab_size")
        self.target_model = target_model
        self.draft_model = draft_model

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        config: Optional[InferenceConfig] = None,
        max_new_tokens: Optional[int] = None,
        gamma: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        eos_token_id: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> SpeculativeResult:
        """Generate continuation tokens via speculative draft-and-verify decoding.

        Note: Operates on batch_size == 1 so variable rejection lengths across sequences
        do not require ragged KV-cache padding.
        """
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError(
                f"SpeculativeDecoder requires prompt_ids of shape (1, seq_len), got {prompt_ids.shape}"
            )
        if prompt_ids.shape[1] == 0:
            raise ValueError("prompt_ids must contain at least 1 token")

        cfg = config or InferenceConfig()
        max_new = cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        gamma_val = cfg.speculative_gamma if gamma is None else gamma
        if gamma_val < 1:
            raise ValueError(f"speculative gamma must be >= 1, got {gamma_val}")
        temp = cfg.temperature if temperature is None else temperature
        k_val = cfg.top_k if top_k is None else top_k
        p_val = cfg.top_p if top_p is None else top_p
        sample_flag = cfg.do_sample if do_sample is None else do_sample
        eos_id = cfg.eos_token_id if eos_token_id is None else eos_token_id

        was_target_training = self.target_model.training
        was_draft_training = self.draft_model.training
        self.target_model.eval()
        self.draft_model.eval()

        target_device = next(self.target_model.parameters()).device
        draft_device = next(self.draft_model.parameters()).device

        tokens = prompt_ids.to(target_device)
        prompt_len = tokens.shape[1]
        max_context = min(
            self.target_model.config.max_seq_len,
            self.draft_model.config.max_seq_len,
        )
        max_allowed_new = min(max_new, max(0, max_context - prompt_len))

        target_cache: KVCache = self.target_model.create_kv_cache()
        draft_cache: KVCache = self.draft_model.create_kv_cache()

        t0 = time.perf_counter()

        # Invariant at the start of every speculative round:
        # Both target_cache and draft_cache hold KV states for tokens[:, :-1].
        if prompt_len > 1:
            prefix = tokens[:, :-1]
            self.target_model(prefix, kv_cache=target_cache)
            self.draft_model(prefix.to(draft_device), kv_cache=draft_cache)

        generated_tokens: List[int] = []
        total_drafted = 0
        total_accepted = 0
        num_target_calls = 0

        while len(generated_tokens) < max_allowed_new:
            remaining = max_allowed_new - len(generated_tokens)
            cur_gamma = min(gamma_val, remaining)

            # 1. Draft `cur_gamma` candidate tokens autoregressively
            draft_ids: List[int] = []
            draft_probs: List[torch.Tensor] = []
            cur_tok = tokens[:, -1:].to(draft_device)

            for _ in range(cur_gamma):
                d_out = self.draft_model(cur_tok, kv_cache=draft_cache)
                d_logits = d_out.logits[:, -1, :]
                if sample_flag and temp > 0.0:
                    q_probs = compute_sampling_probs(
                        d_logits, temperature=temp, top_k=k_val, top_p=p_val
                    )
                    next_d = torch.multinomial(q_probs, num_samples=1, generator=generator).squeeze(-1)
                else:
                    q_probs = torch.zeros_like(d_logits, dtype=torch.float32)
                    next_d = torch.argmax(d_logits, dim=-1)
                    q_probs.scatter_(1, next_d.unsqueeze(-1), 1.0)

                tok_id = int(next_d.item())
                draft_ids.append(tok_id)
                draft_probs.append(q_probs.squeeze(0).to(target_device))
                cur_tok = next_d.unsqueeze(0).to(draft_device)

            total_drafted += cur_gamma

            # 2. Verify `[tokens[-1], d_0, ..., d_{cur_gamma-1}]` in ONE target forward pass
            draft_tensor = torch.tensor([draft_ids], dtype=tokens.dtype, device=target_device)
            verify_input = torch.cat([tokens[:, -1:], draft_tensor], dim=1)
            t_out = self.target_model(verify_input, kv_cache=target_cache)
            num_target_calls += 1
            t_logits = t_out.logits  # (1, cur_gamma + 1, vocab)

            # 3. Acceptance / rejection sampling
            accepted_round: List[int] = []
            all_accepted = True

            for i in range(cur_gamma):
                d_tok = draft_ids[i]
                step_t_logits = t_logits[:, i, :]

                if not sample_flag or temp <= 0.0:
                    target_argmax = int(torch.argmax(step_t_logits, dim=-1).item())
                    if d_tok == target_argmax:
                        accepted_round.append(d_tok)
                        total_accepted += 1
                    else:
                        accepted_round.append(target_argmax)
                        all_accepted = False
                        break
                else:
                    p_probs = compute_sampling_probs(
                        step_t_logits, temperature=temp, top_k=k_val, top_p=p_val
                    ).squeeze(0)
                    q_probs = draft_probs[i]
                    p_val_tok = float(p_probs[d_tok].item())
                    q_val_tok = float(q_probs[d_tok].item())
                    accept_prob = min(1.0, p_val_tok / max(q_val_tok, 1e-12))

                    u = float(torch.rand((), generator=generator, device=target_device).item())
                    if u <= accept_prob:
                        accepted_round.append(d_tok)
                        total_accepted += 1
                    else:
                        residual = torch.clamp(p_probs - q_probs, min=0.0)
                        residual_sum = float(residual.sum().item())
                        if residual_sum > 1e-12:
                            residual = residual / residual_sum
                        else:
                            residual = p_probs
                        corrected = int(
                            torch.multinomial(residual, num_samples=1, generator=generator).item()
                        )
                        accepted_round.append(corrected)
                        all_accepted = False
                        break

                if eos_id is not None and accepted_round[-1] == eos_id:
                    all_accepted = False
                    break

            # 4. Bonus token if all draft tokens were accepted and budget allows
            hit_eos = eos_id is not None and accepted_round[-1] == eos_id
            if all_accepted and not hit_eos and (len(generated_tokens) + len(accepted_round) < max_allowed_new):
                bonus_logits = t_logits[:, cur_gamma, :]
                bonus_tok = int(
                    sample_next_token(
                        bonus_logits,
                        temperature=temp,
                        top_k=k_val,
                        top_p=p_val,
                        do_sample=sample_flag,
                        generator=generator,
                    ).item()
                )
                accepted_round.append(bonus_tok)

            # Truncate accepted_round to remaining budget and stop at EOS if present
            accepted_round = accepted_round[:remaining]
            if eos_id is not None and eos_id in accepted_round:
                eos_idx = accepted_round.index(eos_id)
                accepted_round = accepted_round[: eos_idx + 1]
                hit_eos = True

            generated_tokens.extend(accepted_round)
            new_tensor = torch.tensor([accepted_round], dtype=tokens.dtype, device=target_device)
            tokens = torch.cat([tokens, new_tensor], dim=1)

            # 5. Synchronize both KV caches to hold exactly `tokens[:, :-1]`
            target_keep_len = tokens.shape[1] - 1
            target_cache.truncate(target_keep_len)
            draft_cache.truncate(target_keep_len)

            # If all `cur_gamma` draft tokens + bonus were accepted, draft_cache is 1 token behind
            if draft_cache.seq_len(0) < target_keep_len:
                missing_tok = tokens[:, draft_cache.seq_len(0) : target_keep_len].to(draft_device)
                self.draft_model(missing_tok, kv_cache=draft_cache)

            if hit_eos:
                break

        t_end = time.perf_counter()

        if was_target_training:
            self.target_model.train()
        if was_draft_training:
            self.draft_model.train()

        gen_tensor = torch.tensor([generated_tokens], dtype=tokens.dtype, device=target_device)
        elapsed = t_end - t0
        n_gen = len(generated_tokens)
        return SpeculativeResult(
            sequences=tokens,
            generated_tokens=gen_tensor,
            num_generated_tokens=n_gen,
            total_draft_tokens=total_drafted,
            accepted_draft_tokens=total_accepted,
            acceptance_rate=(total_accepted / total_drafted) if total_drafted > 0 else 0.0,
            num_target_forward_calls=num_target_calls,
            total_time_s=elapsed,
            tokens_per_second=n_gen / max(elapsed, 1e-9),
        )
