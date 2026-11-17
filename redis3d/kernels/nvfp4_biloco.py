"""Hybrid NVFP4 E2M1 + SVDQuant r16 + Binary ±1 r256 BiLoCo (roadfinal.md §3.5, §7 Phase 6).

Weight Approximation:
  W ≈ Q_NVFP4(W_res) + U_16 @ V_16^T + B_{±1, 256} @ diag(D_scale) @ C_{±1, 256}^T

Storage:
  - NVFP4 E2M1 weights: 4-bit nibbles packed 2-per-uint8 -> [out, in // 2] uint8
  - Group-16 micro-scales: FP8/FP16 per 16-element block -> [out, in // 16] float8_e4m3fn (or float16)
  - FP16 Top-16 SVD adapter: U_16 [out, 16], V_16 [16, in]
  - Binary ±1 Rank-256 BiLoCo spectral tail:
      B_pm1 packed into uint64 words [out, 256 // 64]
      C_pm1 packed into uint64 words [in,  256 // 64]
      D_scale [256] float16
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Exact E2M1 representable positive values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
_E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
# Full 16-entry LUT indexed by 4-bit nibble: bit3=sign, bits[2:0]=magnitude index
_E2M1_LUT = torch.cat([_E2M1_POS, -_E2M1_POS], dim=0)


def quantize_e2m1_block16(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight tensor [out, in] into packed E2M1 uint8 + group-16 scales.

    Returns:
        packed: [out, ceil(in/2)] uint8
        scales: [out, ceil(in/16)] float16 (or float8_e4m3fn-quantized float16)
    """
    out_f, in_f = w.shape
    pad_in = (16 - (in_f % 16)) % 16
    w_pad = F.pad(w.float(), (0, pad_in)) if pad_in > 0 else w.float()
    in_padded = w_pad.shape[1]

    blocks = w_pad.view(out_f, in_padded // 16, 16)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scales = (amax / 6.0).squeeze(-1)  # [out, n_blocks]

    # Quantize scales to FP8 E4M3 precision if supported, keeping tensor dtype compact
    if hasattr(torch, "float8_e4m3fn"):
        scales_fp8 = scales.clamp(max=448.0).to(torch.float8_e4m3fn)
        scales_f = scales_fp8.float().clamp(min=1e-6)
    else:
        scales_fp8 = scales.half()
        scales_f = scales_fp8.float().clamp(min=1e-6)

    w_norm = blocks / scales_f.unsqueeze(-1)  # in [-6, 6]
    pos_lut = _E2M1_POS.to(w.device)

    # Nearest magnitude index in [0..7]
    abs_w = w_norm.abs().unsqueeze(-1)  # [out, n_blk, 16, 1]
    mag_idx = torch.argmin((abs_w - pos_lut).abs(), dim=-1)  # [out, n_blk, 16]
    sign_bit = (w_norm < 0).to(torch.uint8) * 8
    nibbles = (sign_bit | mag_idx.to(torch.uint8)).view(out_f, in_padded)

    # Pack two 4-bit nibbles per uint8
    low = nibbles[:, 0::2]
    high = nibbles[:, 1::2]
    packed = (low & 0x0F) | ((high & 0x0F) << 4)
    return packed, scales_fp8


def dequantize_e2m1_block16(
    packed: torch.Tensor,
    scales: torch.Tensor,
    in_features: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize packed E2M1 uint8 + group-16 scales back to [out, in_features]."""
    out_f = packed.shape[0]
    low = (packed & 0x0F).to(torch.long)
    high = ((packed >> 4) & 0x0F).to(torch.long)
    nibbles = torch.stack([low, high], dim=-1).view(out_f, -1)

    lut = _E2M1_LUT.to(device=packed.device, dtype=torch.float32)
    vals = lut[nibbles]  # [out, in_padded]
    n_blocks = scales.shape[1]
    vals = vals.view(out_f, n_blocks, 16) * scales.float().unsqueeze(-1)
    return vals.view(out_f, -1)[:, :in_features].to(dtype)


def pack_binary_pm1_uint64(mat: torch.Tensor) -> torch.Tensor:
    """Pack a {+1, -1} matrix [N, R] (R multiple of 64) into uint64 bitmasks [N, R // 64]."""
    n, r = mat.shape
    assert r % 64 == 0, f"Rank {r} must be a multiple of 64 for uint64 packing."
    bits = (mat > 0).to(torch.int64).view(n, r // 64, 64)
    shifts = torch.arange(64, device=mat.device, dtype=torch.int64)
    # Note: bit 63 is sign bit in int64, which is exact bit-pattern preserved in view(uint64)
    packed_i64 = (bits << shifts).sum(dim=-1)
    return packed_i64.view(torch.uint64)


def unpack_binary_pm1_uint64(packed: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Unpack uint64 bitmasks [N, words] back to {+1.0, -1.0} tensor [N, words * 64]."""
    n, words = packed.shape
    packed_i64 = packed.view(torch.int64).unsqueeze(-1)  # [N, words, 1]
    shifts = torch.arange(64, device=packed.device, dtype=torch.int64)
    bits = (packed_i64 >> shifts) & 1
    pm1 = bits.to(dtype) * 2.0 - 1.0
    return pm1.view(n, words * 64)


def decompose_nvfp4_biloco(
    weight: torch.Tensor,
    svd_rank: int = 16,
    biloco_rank: int = 256,
    act_scales: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Decompose weight [out, in] into NVFP4 E2M1 + FP16 SVD r16 + Binary ±1 BiLoCo r256."""
    w = weight.detach().float()
    out_f, in_f = w.shape

    # Optional AWQ channel scaling
    if act_scales is not None:
        s = act_scales.detach().float().clamp(min=1e-4)
        s = s / s.mean()
        w_scaled = w * s.unsqueeze(0)
    else:
        s = None
        w_scaled = w

    # 1. FP16 Top-16 SVD Adapter (absorbs channel outliers)
    r_svd = min(svd_rank, out_f, in_f)
    u_full, sig_full, vh_full = torch.linalg.svd(w_scaled, full_matrices=False)
    sqrt_sig = torch.sqrt(sig_full[:r_svd])
    u16 = u_full[:, :r_svd] * sqrt_sig.unsqueeze(0)
    v16 = sqrt_sig.unsqueeze(1) * vh_full[:r_svd, :]
    w_res = w_scaled - (u16 @ v16)

    # 2. Binary ±1 Rank-256 BiLoCo Spectral Tail
    # Extract next spectral modes from w_res and binarize their singular vectors
    r_eff = min(biloco_rank, out_f, in_f)
    if r_eff > 0:
        u_b, sig_b, vh_b = torch.linalg.svd(w_res, full_matrices=False)
        b_mat = torch.where(u_b[:, :r_eff] >= 0, 1.0, -1.0)
        c_mat = torch.where(vh_b[:r_eff, :].T >= 0, 1.0, -1.0)
        # Optimal least-squares diagonal scale per rank-1 binary outer product
        # d_k ≈ (u_k^T W_res v_k) / (||b_k||^2 ||c_k||^2) = (b_k^T W_res c_k) / (out_f * in_f)
        # Project sequentially in blocks or jointly via inner product
        d_scale = torch.sum((b_mat.T @ w_res) * c_mat.T, dim=-1) / float(out_f * in_f)
        if r_eff < biloco_rank:
            b_mat = F.pad(b_mat, (0, biloco_rank - r_eff), value=1.0)
            c_mat = F.pad(c_mat, (0, biloco_rank - r_eff), value=1.0)
            d_scale = F.pad(d_scale, (0, biloco_rank - r_eff), value=0.0)
        w_res = w_res - (b_mat * d_scale.unsqueeze(0)) @ c_mat.T
    else:
        b_mat = torch.ones((out_f, biloco_rank), device=w.device)
        c_mat = torch.ones((in_f, biloco_rank), device=w.device)
        d_scale = torch.zeros((biloco_rank,), device=w.device)

    # Unscale back if AWQ act_scales was applied
    if s is not None:
        inv_s = 1.0 / s
        u16_out = u16
        v16_out = v16 * inv_s.unsqueeze(0)
        w_res_unscaled = w_res * inv_s.unsqueeze(0)
        # Keep BiLoCo in unscaled space by folding inv_s into NVFP4 residual
        w_biloco_approx = (b_mat * d_scale.unsqueeze(0)) @ c_mat.T * inv_s.unsqueeze(0)
        # Re-fit d_scale cleanly on unscaled residual after SVD
        w_after_svd = w - (u16_out @ v16_out)
        d_scale = torch.sum((b_mat.T @ w_after_svd) * c_mat.T, dim=-1) / float(out_f * in_f)
        w_res_unscaled = w_after_svd - (b_mat * d_scale.unsqueeze(0)) @ c_mat.T
    else:
        u16_out = u16
        v16_out = v16
        w_res_unscaled = w_res

    # 3. Quantize remaining residual to E2M1 NVFP4 + Group-16 scales
    w_packed, w_scales = quantize_e2m1_block16(w_res_unscaled)

    return {
        "w_packed": w_packed,
        "w_scales": w_scales,
        "u16": u16_out.half(),
        "v16": v16_out.half(),
        "b_packed": pack_binary_pm1_uint64(b_mat),
        "c_packed": pack_binary_pm1_uint64(c_mat),
        "d_scale": d_scale.half(),
    }


class NVFP4BiLoCoLinear(nn.Module):
    """Drop-in dense Linear layer using NVFP4 E2M1 + SVDQuant r16 + Binary r256 BiLoCo."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        svd_rank: int = 16,
        biloco_rank: int = 256,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.svd_rank = svd_rank
        self.biloco_rank = biloco_rank

        in_padded = ((in_features + 15) // 16) * 16
        self.register_buffer("w_packed", torch.zeros(out_features, in_padded // 2, dtype=torch.uint8))
        scale_dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16
        self.register_buffer("w_scales", torch.zeros(out_features, in_padded // 16, dtype=scale_dtype))
        self.register_buffer("u16", torch.zeros(out_features, min(svd_rank, out_features, in_features), dtype=torch.float16))
        self.register_buffer("v16", torch.zeros(min(svd_rank, out_features, in_features), in_features, dtype=torch.float16))
        self.register_buffer("b_packed", torch.zeros(out_features, biloco_rank // 64, dtype=torch.uint64))
        self.register_buffer("c_packed", torch.zeros(in_features, biloco_rank // 64, dtype=torch.uint64))
        self.register_buffer("d_scale", torch.zeros(biloco_rank, dtype=torch.float16))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        svd_rank: int = 16,
        biloco_rank: int = 256,
        act_scales: Optional[torch.Tensor] = None,
    ) -> "NVFP4BiLoCoLinear":
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            svd_rank=svd_rank,
            biloco_rank=biloco_rank,
        ).to(linear.weight.device)

        parts = decompose_nvfp4_biloco(
            linear.weight.data,
            svd_rank=svd_rank,
            biloco_rank=biloco_rank,
            act_scales=act_scales,
        )
        layer.w_packed.copy_(parts["w_packed"])
        layer.w_scales.copy_(parts["w_scales"])
        layer.u16.copy_(parts["u16"])
        layer.v16.copy_(parts["v16"])
        layer.b_packed.copy_(parts["b_packed"])
        layer.c_packed.copy_(parts["c_packed"])
        layer.d_scale.copy_(parts["d_scale"])
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data.half())
        return layer

    def reconstruct_weight(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Reconstruct effective dense weight matrix [out_features, in_features]."""
        w_nvfp4 = dequantize_e2m1_block16(self.w_packed, self.w_scales, self.in_features, dtype=dtype)
        w_svd = (self.u16.to(dtype) @ self.v16.to(dtype))
        b_pm1 = unpack_binary_pm1_uint64(self.b_packed, dtype=dtype)
        c_pm1 = unpack_binary_pm1_uint64(self.c_packed, dtype=dtype)
        w_biloco = (b_pm1 * self.d_scale.to(dtype).unsqueeze(0)) @ c_pm1.T
        return w_nvfp4 + w_svd + w_biloco

    def storage_bits_per_param(self) -> float:
        """Compute total stored bits per parameter (NVFP4 + scales + SVD16 + BiLoCo256)."""
        n_params = self.out_features * self.in_features
        total_bits = (
            self.w_packed.numel() * 8
            + self.w_scales.numel() * (8 if self.w_scales.dtype != torch.float16 else 16)
            + self.u16.numel() * 16
            + self.v16.numel() * 16
            + self.b_packed.numel() * 64
            + self.c_packed.numel() * 64
            + self.d_scale.numel() * 16
        )
        return total_bits / max(1, n_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from redis3d.kernels.triton_nvfp4_linear import fused_nvfp4_biloco_linear

        return fused_nvfp4_biloco_linear(x, self)
