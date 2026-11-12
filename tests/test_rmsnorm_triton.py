"""Tests for RMSNorm PyTorch reference and custom Triton CUDA kernel + fallback."""

import pytest
import torch

from sciminilm.layers.rmsnorm import (
    RMSNorm,
    is_triton_available,
    rmsnorm_pytorch,
    rmsnorm_triton,
)


def test_rmsnorm_pytorch_matches_manual():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 64, requires_grad=True)
    norm = RMSNorm(64, eps=1e-6, use_triton=False)

    y = norm(x)
    manual = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * norm.weight
    assert torch.allclose(y, manual, atol=1e-6)

    y.sum().backward()
    assert x.grad is not None and norm.weight.grad is not None


def test_rmsnorm_cpu_graceful_fallback():
    x = torch.randn(3, 8, 128)
    w = torch.ones(128)
    y_triton_call = rmsnorm_triton(x, w, eps=1e-6)
    y_ref = rmsnorm_pytorch(x, w, eps=1e-6)
    assert torch.allclose(y_triton_call, y_ref, atol=1e-7)

    norm = RMSNorm(128, use_triton=True)
    assert torch.allclose(norm(x), y_ref, atol=1e-7)


@pytest.mark.parametrize("dim", [64, 300, 640])
def test_rmsnorm_triton_cuda_matches_pytorch(dim: int):
    if not is_triton_available():
        pytest.skip("Triton/CUDA not available on this machine")

    torch.manual_seed(42)
    x_ref = torch.randn(4, 17, dim, device="cuda", dtype=torch.float32, requires_grad=True)
    w_ref = torch.randn(dim, device="cuda", dtype=torch.float32, requires_grad=True)

    x_tri = x_ref.detach().clone().requires_grad_(True)
    w_tri = w_ref.detach().clone().requires_grad_(True)

    y_ref = rmsnorm_pytorch(x_ref, w_ref, eps=1e-6)
    y_tri = rmsnorm_triton(x_tri, w_tri, eps=1e-6)
    assert torch.allclose(y_tri, y_ref, atol=1e-5, rtol=1e-4)

    grad_out = torch.randn_like(y_ref)
    y_ref.backward(grad_out)
    y_tri.backward(grad_out)

    assert x_tri.grad is not None and x_ref.grad is not None
    assert w_tri.grad is not None and w_ref.grad is not None
    assert torch.allclose(x_tri.grad, x_ref.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(w_tri.grad, w_ref.grad, atol=1e-4, rtol=1e-3)
