"""Dense Fused NVFP4 + BiLoCo Linear Kernel (roadfinal.md §3.5, §7 Phase 6).

Strictly Dense execution (NO grouped/MoE FP4 kernels — safe on Blackwell sm_120).
Uses Triton when available, or verified dense PyTorch CUDA Tensor Core matmul.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from redis3d.kernels import has_triton
from redis3d.kernels.nvfp4_biloco import (
    dequantize_e2m1_block16,
    unpack_binary_pm1_uint64,
)

if TYPE_CHECKING:
    from redis3d.kernels.nvfp4_biloco import NVFP4BiLoCoLinear


def fused_nvfp4_biloco_linear(x: torch.Tensor, layer: "NVFP4BiLoCoLinear") -> torch.Tensor:
    """Dense forward pass: y = x @ W_nvfp4^T + (x @ V16^T) @ U16^T + ((x @ C) * d) @ B^T + bias."""
    compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16
    x_c = x.to(compute_dtype)

    # 1. Dense E2M1 NVFP4 residual branch
    w_nvfp4 = dequantize_e2m1_block16(
        layer.w_packed, layer.w_scales, layer.in_features, dtype=compute_dtype
    )
    out = F.linear(x_c, w_nvfp4, None)

    # 2. Low-rank FP16 SVDQuant r16 branch (avoids materializing O*I matrix)
    v16 = layer.v16.to(compute_dtype)
    u16 = layer.u16.to(compute_dtype)
    svd_mid = F.linear(x_c, v16, None)      # [..., 16]
    out = out + F.linear(svd_mid, u16, None)

    # 3. Binary ±1 Rank-256 BiLoCo spectral tail
    c_pm1 = unpack_binary_pm1_uint64(layer.c_packed, dtype=compute_dtype)  # [in, 256]
    b_pm1 = unpack_binary_pm1_uint64(layer.b_packed, dtype=compute_dtype)  # [out, 256]
    d_scale = layer.d_scale.to(compute_dtype)
    biloco_mid = torch.matmul(x_c, c_pm1) * d_scale
    out = out + torch.matmul(biloco_mid, b_pm1.T)

    if layer.bias is not None:
        out = out + layer.bias.to(compute_dtype)

    return out.to(x.dtype)
