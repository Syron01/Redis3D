"""Phase 6 Acceptance Gate: Blackwell sm_120 NVFP4 + BiLoCo-SVDQuant (roadfinal.md §7 Phase 6).

Contract:
  1. Quantized layer cosine similarity >= 0.994 vs FP16 reference (and BiLoCo + SVDQuant
     strictly beats plain FP4 and plain SVDQuant r16).
  2. Packed uint8 NVFP4 + uint64 BiLoCo storage achieves compact bits/param.
  3. Sensitive-layer shielding keeps coord/time/in-out heads in FP16/FP32 while
     quantizing backbone Linears on Stage2LatentMeanFlow.
  4. Forward speed & sm_120 execution complete without illegal instruction or NaNs.
"""

from __future__ import annotations

import copy
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from redis3d.core.hardware_guard import VRAMBudgetGuard
from redis3d.kernels.nvfp4_biloco import (
    NVFP4BiLoCoLinear,
    dequantize_e2m1_block16,
    quantize_e2m1_block16,
)
from redis3d.kernels.triton_sparse_attn import sparse_window_attention_qat
from redis3d.models.latent_meanflow import Stage2LatentMeanFlow
from redis3d.training.quantize_nvfp4_biloco import (
    collect_activation_scales,
    quantize_model_nvfp4_biloco,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real GPU required")
DEV = "cuda"


def test_nvfp4_biloco_cosine_similarity_beats_plain_fp4_and_svd16():
    """Verify NVFP4 + SVDQuant r16 + BiLoCo r256 achieves cosine >= 0.994 and beats baselines."""
    torch.manual_seed(42)
    in_f, out_f = 1024, 1024
    linear = nn.Linear(in_f, out_f, bias=True).to(DEV)
    # Inject structured spectral + outlier channels realistic for 3D neural weights
    with torch.no_grad():
        u = torch.randn(out_f, 48, device=DEV)
        v = torch.randn(48, in_f, device=DEV)
        w = (u @ v) / (48.0 ** 0.5) + 0.15 * torch.randn(out_f, in_f, device=DEV)
        w[:, :4] *= 5.0  # channel outliers
        linear.weight.copy_(w)

    x = torch.randn(8, 64, in_f, device=DEV, dtype=torch.float16)
    with torch.no_grad():
        y_ref = F.linear(x, linear.weight.half(), linear.bias.half()).float()

    # 1. Plain FP4 (svd_rank=0, biloco_rank=0)
    q_fp4 = NVFP4BiLoCoLinear.from_linear(linear, svd_rank=0, biloco_rank=0)
    y_fp4 = q_fp4(x).float()

    # 2. Plain SVDQuant r16 (biloco_rank=0)
    q_svd16 = NVFP4BiLoCoLinear.from_linear(linear, svd_rank=16, biloco_rank=0)
    y_svd16 = q_svd16(x).float()

    # 3. Full Hybrid NVFP4 + SVDQuant r16 + Binary r256 BiLoCo
    act_scales = x.float().abs().mean(dim=(0, 1))
    q_biloco = NVFP4BiLoCoLinear.from_linear(
        linear, svd_rank=16, biloco_rank=256, act_scales=act_scales
    )
    y_biloco = q_biloco(x).float()

    cos_fp4 = F.cosine_similarity(y_fp4.flatten(), y_ref.flatten(), dim=0).item()
    cos_svd16 = F.cosine_similarity(y_svd16.flatten(), y_ref.flatten(), dim=0).item()
    cos_biloco = F.cosine_similarity(y_biloco.flatten(), y_ref.flatten(), dim=0).item()

    bpp = q_biloco.storage_bits_per_param()
    print(
        f"\n[NVFP4+BiLoCo] CosSim: PlainFP4={cos_fp4:.5f} | SVD16={cos_svd16:.5f} | "
        f"BiLoCo+SVD16={cos_biloco:.5f} (SLA >= 0.994) | Storage={bpp:.2f} bits/param"
    )
    assert cos_biloco >= 0.994, f"Cosine similarity {cos_biloco:.5f} < 0.994 SLA!"
    assert cos_biloco > cos_fp4, "BiLoCo+SVD16 must beat plain FP4!"
    assert cos_biloco > cos_svd16, "BiLoCo+SVD16 must beat plain SVD16!"
    assert bpp < 6.0, f"Storage {bpp:.2f} bits/param is too large!"


def test_sensitive_layer_shielding_and_sm120_forward_benchmark():
    """Verify sensitive-layer shielding on Stage2LatentMeanFlow and sm_120 forward execution."""
    torch.manual_seed(0)
    model = Stage2LatentMeanFlow(
        latent_dim=24,
        model_dim=256,
        cond_dim=256,
        num_groups=2,
        num_heads=8,
        window_size=64,
        num_registers=16,
    ).to(DEV).eval()

    z0 = torch.randn(2, 256, 24, device=DEV)
    cond = torch.randn(2, 256, device=DEV)

    with torch.no_grad():
        ref_out = model.sample_1nfe(z0, cond)

    def _calib_fn(m):
        m.sample_1nfe(z0, cond)

    act_scales = collect_activation_scales(model, _calib_fn)
    q_model, stats = quantize_model_nvfp4_biloco(
        copy.deepcopy(model), act_scales=act_scales, svd_rank=16, biloco_rank=256
    )
    q_model.eval()

    # Verify shielded layers stayed nn.Linear while backbone layers converted
    assert isinstance(q_model.in_proj, nn.Linear)
    assert isinstance(q_model.out_proj, nn.Linear)
    assert stats["quantized"] > 0 and stats["shielded"] > 0

    with VRAMBudgetGuard(max_mb=2048):
        with torch.no_grad():
            for _ in range(2):
                _ = q_model.sample_1nfe(z0, cond)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            q_out = q_model.sample_1nfe(z0, cond)
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0

    assert torch.isfinite(q_out).all(), "NaN/Inf detected in sm_120 quantized forward!"
    cos = F.cosine_similarity(q_out.flatten(), ref_out.flatten(), dim=0).item()
    print(
        f"\n[sm_120 Quantized Stage2] quantized={stats['quantized']} shielded={stats['shielded']} | "
        f"1-NFE latency={dt_ms:.2f} ms | CosSim vs FP32={cos:.4f}"
    )
    assert cos >= 0.90


def test_sparse_window_attention_qat_stability():
    """Verify Attn-QAT symmetric QK-RMSNorm produces finite, well-scaled outputs in FP16."""
    q = torch.randn(2, 8, 256, 32, device=DEV, dtype=torch.float16) * 20.0
    k = torch.randn(2, 8, 256, 32, device=DEV, dtype=torch.float16) * 20.0
    v = torch.randn(2, 8, 256, 32, device=DEV, dtype=torch.float16)

    out = sparse_window_attention_qat(q, k, v, window_size=64, apply_qk_norm=True)
    assert out.shape == v.shape
    assert torch.isfinite(out).all()
