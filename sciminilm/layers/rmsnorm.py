"""RMSNorm layer with custom Triton CUDA kernel and graceful PyTorch fallback."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    """Return True if Triton is importable and CUDA is available."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


def rmsnorm_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference pure-PyTorch RMSNorm computed in fp32 for numerical stability."""
    input_dtype = x.dtype
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    return (x_normed * weight.float()).to(input_dtype)


if _TRITON_AVAILABLE:

    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr,
        W_ptr,
        Y_ptr,
        Rstd_ptr,
        stride_x_row,
        stride_y_row,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = tl.rsqrt(mean_sq + eps)
        tl.store(Rstd_ptr + row, rstd)

        y = x * rstd * w
        tl.store(Y_ptr + row * stride_y_row + cols, y, mask=mask)

    @triton.jit
    def _rmsnorm_bwd_dx_kernel(
        DX_ptr,
        DY_ptr,
        X_ptr,
        W_ptr,
        Rstd_ptr,
        stride_dx_row,
        stride_dy_row,
        stride_x_row,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd_ptr + row).to(tl.float32)

        x_hat = x * rstd
        wdy = w * dy
        mean_wdy_xhat = tl.sum(wdy * x_hat, axis=0) / N
        dx = rstd * (wdy - x_hat * mean_wdy_xhat)
        tl.store(DX_ptr + row * stride_dx_row + cols, dx, mask=mask)


class _TritonRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        orig_shape = x.shape
        N = orig_shape[-1]
        x_2d = x.contiguous().view(-1, N)
        M = x_2d.shape[0]

        y = torch.empty_like(x_2d)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        BLOCK_SIZE = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_SIZE <= 1024 else 8

        _rmsnorm_fwd_kernel[(M,)](
            x_2d,
            weight,
            y,
            rstd,
            x_2d.stride(0),
            y.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.orig_shape = orig_shape
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        return y.view(*orig_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_2d, weight, rstd = ctx.saved_tensors
        N = x_2d.shape[-1]
        M = x_2d.shape[0]
        dy_2d = dy.contiguous().view(-1, N)
        dx = torch.empty_like(x_2d)

        _rmsnorm_bwd_dx_kernel[(M,)](
            dx,
            dy_2d,
            x_2d,
            weight,
            rstd,
            dx.stride(0),
            dy_2d.stride(0),
            x_2d.stride(0),
            N,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )

        x_hat = x_2d.float() * rstd.unsqueeze(1)
        dweight = (dy_2d.float() * x_hat).sum(dim=0).to(weight.dtype)
        return dx.view(*ctx.orig_shape), dweight, None


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply Triton RMSNorm if CUDA+Triton are available and N fits, else fallback."""
    if not _TRITON_AVAILABLE or not x.is_cuda:
        return rmsnorm_pytorch(x, weight, eps)
    # Triton blocks above 65536 elements per row can exceed shared memory limits;
    # standard LLM hidden sizes (64..8192) fit comfortably.
    if x.shape[-1] > 32768:
        return rmsnorm_pytorch(x, weight, eps)
    return _TritonRMSNormFn.apply(x, weight, eps)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with optional Triton CUDA acceleration."""

    def __init__(self, dim: int, eps: float = 1e-6, use_triton: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.use_triton = use_triton
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton and _TRITON_AVAILABLE and x.is_cuda and self.dim <= 32768:
            return rmsnorm_triton(x, self.weight, self.eps)
        return rmsnorm_pytorch(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, use_triton={self.use_triton}"
