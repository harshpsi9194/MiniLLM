"""INT8 (per-channel) and INT4 (group-wise packed) weight quantization primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class QuantizedTensor:
    """Container for quantized weight tensor and its quantization metadata."""

    qdata: torch.Tensor
    scales: torch.Tensor
    zero_points: Optional[torch.Tensor]
    bits: int
    group_size: int
    symmetric: bool
    orig_shape: Tuple[int, ...]
    padded_in_features: int


def quantize_int8_per_channel(
    weight: torch.Tensor,
    symmetric: bool = True,
) -> QuantizedTensor:
    """Quantize a 2D weight tensor (out_features, in_features) to INT8 per output channel."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape {tuple(weight.shape)}")

    w_fp32 = weight.detach().float()
    out_features, in_features = w_fp32.shape

    if symmetric:
        abs_max = w_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scales = (abs_max / 127.0).to(torch.float16)
        qdata = torch.round(w_fp32 / scales.float()).clamp(-127, 127).to(torch.int8)
        zero_points = None
    else:
        w_min = w_fp32.amin(dim=-1, keepdim=True)
        w_max = w_fp32.amax(dim=-1, keepdim=True)
        scales = ((w_max - w_min) / 255.0).clamp(min=1e-8).to(torch.float16)
        zp = torch.round(-128.0 - w_min / scales.float()).clamp(-128, 127).to(torch.int8)
        qdata = torch.round(w_fp32 / scales.float() + zp.float()).clamp(-128, 127).to(torch.int8)
        zero_points = zp

    return QuantizedTensor(
        qdata=qdata,
        scales=scales,
        zero_points=zero_points,
        bits=8,
        group_size=in_features,
        symmetric=symmetric,
        orig_shape=(out_features, in_features),
        padded_in_features=in_features,
    )


def dequantize_int8_per_channel(
    qt: QuantizedTensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize an INT8 per-channel QuantizedTensor back to floating point."""
    scales = qt.scales.float()
    if qt.zero_points is None:
        w = qt.qdata.float() * scales
    else:
        w = (qt.qdata.float() - qt.zero_points.float()) * scales
    return w.to(dtype)


def quantize_int4_grouped(
    weight: torch.Tensor,
    group_size: int = 64,
    symmetric: bool = True,
) -> QuantizedTensor:
    """Quantize a 2D weight tensor to packed INT4 (2 values per uint8) per group.

    In symmetric mode, values are quantized to signed [-7, 7] and stored shifted by +8
    as 4-bit unsigned integers [1, 15]. In asymmetric mode, values are quantized to [0, 15].
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape {tuple(weight.shape)}")
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError(f"group_size must be a positive even integer, got {group_size}")

    w_fp32 = weight.detach().float()
    out_features, in_features = w_fp32.shape

    rem = in_features % group_size
    if rem != 0:
        pad_cols = group_size - rem
        w_fp32 = torch.nn.functional.pad(w_fp32, (0, pad_cols), value=0.0)
    padded_in_features = w_fp32.shape[1]
    n_groups = padded_in_features // group_size

    w_grouped = w_fp32.view(out_features, n_groups, group_size)

    if symmetric:
        abs_max = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scales = (abs_max / 7.0).to(torch.float16)
        q_signed = torch.round(w_grouped / scales.float()).clamp(-7, 7)
        q_u4 = (q_signed + 8).to(torch.uint8)
        zero_points = None
    else:
        w_min = w_grouped.amin(dim=-1, keepdim=True)
        w_max = w_grouped.amax(dim=-1, keepdim=True)
        scales = ((w_max - w_min) / 15.0).clamp(min=1e-8).to(torch.float16)
        zp = torch.round(-w_min / scales.float()).clamp(0, 15).to(torch.uint8)
        q_u4 = torch.round(w_grouped / scales.float() + zp.float()).clamp(0, 15).to(torch.uint8)
        zero_points = zp

    # Pack pairs of 4-bit values into uint8: low nibble = even index, high nibble = odd index
    q_flat = q_u4.view(out_features, padded_in_features)
    low = q_flat[:, 0::2]
    high = q_flat[:, 1::2]
    packed = (high << 4) | low

    return QuantizedTensor(
        qdata=packed.contiguous(),
        scales=scales.squeeze(-1).contiguous(),  # (out_features, n_groups)
        zero_points=(
            zero_points.squeeze(-1).contiguous() if zero_points is not None else None
        ),
        bits=4,
        group_size=group_size,
        symmetric=symmetric,
        orig_shape=(out_features, in_features),
        padded_in_features=padded_in_features,
    )


def dequantize_int4_grouped(
    qt: QuantizedTensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpack and dequantize an INT4 group-wise QuantizedTensor."""
    out_features, orig_in_features = qt.orig_shape
    packed = qt.qdata
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    unpacked = torch.empty(
        (out_features, qt.padded_in_features),
        dtype=torch.uint8,
        device=packed.device,
    )
    unpacked[:, 0::2] = low
    unpacked[:, 1::2] = high

    n_groups = qt.padded_in_features // qt.group_size
    unpacked_grouped = unpacked.view(out_features, n_groups, qt.group_size).float()
    scales = qt.scales.unsqueeze(-1).float()

    if qt.zero_points is None:
        w_grouped = (unpacked_grouped - 8.0) * scales
    else:
        zp = qt.zero_points.unsqueeze(-1).float()
        w_grouped = (unpacked_grouped - zp) * scales

    w = w_grouped.view(out_features, qt.padded_in_features)[:, :orig_in_features]
    return w.to(dtype)


def quantize_weight(
    weight: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,
    symmetric: bool = True,
) -> QuantizedTensor:
    """Dispatch weight quantization for 8-bit or 4-bit precision."""
    if bits == 8:
        return quantize_int8_per_channel(weight, symmetric=symmetric)
    if bits == 4:
        return quantize_int4_grouped(weight, group_size=group_size, symmetric=symmetric)
    raise ValueError(f"Unsupported quantization bits={bits}; expected 4 or 8")


def dequantize_weight(
    qt: QuantizedTensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dispatch weight dequantization for 8-bit or 4-bit QuantizedTensor."""
    if qt.bits == 8:
        return dequantize_int8_per_channel(qt, dtype=dtype)
    if qt.bits == 4:
        return dequantize_int4_grouped(qt, dtype=dtype)
    raise ValueError(f"Unsupported quantization bits={qt.bits}; expected 4 or 8")
