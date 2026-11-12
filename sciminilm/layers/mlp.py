"""SwiGLU Feed-Forward Network (MLP) for SciMiniLM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP: down_proj(dropout(silu(gate_proj(x)) * up_proj(x)))."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(gated))
