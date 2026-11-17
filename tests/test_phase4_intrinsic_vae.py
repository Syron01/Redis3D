"""Phase 4 Acceptance Gate: PIXL-Intrinsic De-Lighter & Hyperspherical SW-VAE (roadfinal.md §7 Phase 4).

Contract:
  1. PIXL-Intrinsic decomposes synthetic shaded renders to albedo PSNR > 27 dB in < 25 ms.
  2. SW-VAE hyperspherical bottleneck maintains latent norms within 5% of sqrt(24) ≈ 4.899.
  3. CPUAdamW keeps optimizer states in pinned CPU RAM with 0 MB GPU VRAM overhead.
  4. SW-VAE overfits 32 diverse procedural assets on RTX 5060 (< 5.2 GB training VRAM,
     0 MB shared spill) achieving F-Score@0.01 >= 0.990 reconstruction.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from redis3d.core.hardware_guard import VRAMBudgetGuard, shared_gpu_spill_mb
from redis3d.data.procedural_gen import generate_procedural_batch
from redis3d.models.intrinsic_delight import (
    calculate_psnr,
    count_parameters,
    create_pixl_intrinsic,
)
from redis3d.models.sw_vae import (
    LATENT_DIM,
    TARGET_SHELL_RADIUS,
    SparseWaveletVAE,
    compute_fscore_at_threshold,
)
from redis3d.training.cpu_adamw import CPUAdamW
from redis3d.training.train_intrinsic import (
    generate_synthetic_render_batch,
    train_pixl_intrinsic,
)
from redis3d.training.train_vae import prepare_voxel_tensor_batch, train_vae_on_assets

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real GPU required")
DEV = "cuda"


# ============================================================================
# 1. PIXL-Intrinsic De-Lighter Gate Tests
# ============================================================================

def test_pixl_intrinsic_architecture_and_latency():
    """Verify PIXL-Intrinsic parameter budget (~18M) and < 25 ms latency on RTX 5060."""
    model = create_pixl_intrinsic(base_channels=48).to(DEV).eval()
    n_params = count_parameters(model)
    print(f"\n[PIXL-Intrinsic] Parameter count: {n_params:,} (Budget: ~18M)")
    assert 5_000_000 <= n_params <= 25_000_000, f"Unexpected param count: {n_params}"

    # Benchmark latency on 256x256 resolution
    x = torch.rand(1, 3, 256, 256, device=DEV)
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        n_iters = 20
        for _ in range(n_iters):
            _ = model(x)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) / n_iters * 1000.0

    print(f"[PIXL-Intrinsic] Latency: {dt_ms:.2f} ms per 256x256 frame (SLA: < 25.0 ms)")
    assert dt_ms < 25.0, f"Latency {dt_ms:.2f} ms exceeds 25 ms SLA!"


def test_pixl_intrinsic_delight_psnr_convergence():
    """Verify PIXL-Intrinsic separates synthetic shaded renders to albedo PSNR > 27 dB."""
    with VRAMBudgetGuard(max_mb=2048):
        model, final_psnr = train_pixl_intrinsic(
            num_steps=120,
            batch_size=4,
            size=128,
            lr=1.5e-3,
            device=torch.device(DEV),
        )

    print(f"\n[PIXL-Intrinsic] Validation Albedo PSNR: {final_psnr:.2f} dB (SLA: > 27.0 dB)")
    assert final_psnr > 27.0, f"PSNR {final_psnr:.2f} dB did not achieve > 27.0 dB threshold!"


# ============================================================================
# 2. CPUAdamW Optimizer Offload Tests
# ============================================================================

def test_cpu_adamw_optimizer_states_zero_vram():
    """Verify that CPUAdamW stores optimizer state tensors strictly in CPU memory."""
    # Allocate a small model on GPU
    gpu_param = torch.nn.Parameter(torch.randn(1024, 1024, device=DEV))
    optimizer = CPUAdamW([gpu_param], lr=1e-3)

    # Perform optimization step
    loss = (gpu_param ** 2).sum()
    loss.backward()

    vram_before = torch.cuda.memory_allocated()
    optimizer.step()
    vram_after = torch.cuda.memory_allocated()

    # GPU VRAM allocated for optimizer states must be 0
    state = optimizer.state[gpu_param]
    assert state["exp_avg"].device.type == "cpu"
    assert state["exp_avg_sq"].device.type == "cpu"
    assert vram_after <= vram_before + 1024  # zero growth on GPU for states
    print("\n[CPUAdamW] Optimizer state verified in pinned host RAM (0 MB GPU VRAM)")


# ============================================================================
# 3. SW-VAE Hyperspherical Bottleneck & Overfitting Gate Tests
# ============================================================================

def test_hyperspherical_sw_vae_shell_norm():
    """Verify that RMS-Norm Hyperspherical Shell projection guarantees latent norms ≈ sqrt(24)."""
    vae = SparseWaveletVAE(embed_dim=128, latent_dim=LATENT_DIM, num_layers=2).to(DEV).eval()

    B, L = 4, 256
    coords = torch.rand(B, L, 3, device=DEV) * 1.6 - 0.8
    normals = torch.rand(B, L, 2, device=DEV) * 2.0 - 1.0
    pbr = torch.rand(B, L, 7, device=DEV)
    ranks = torch.randint(0, 4, (B, L), device=DEV)

    with torch.no_grad():
        latent = vae.encode(coords, normals, pbr, ranks)

    mean_norm = latent.norms.mean().item()
    target_norm = math.sqrt(LATENT_DIM)  # sqrt(24) ≈ 4.89898
    norm_err = abs(mean_norm - target_norm) / target_norm

    print(f"\n[SW-VAE] Latent shell radius: {mean_norm:.4f} (Target: {target_norm:.4f}, Err: {norm_err:.2%})")
    assert norm_err < 0.05, f"Latent norm error {norm_err:.2%} exceeds 5% bound of sqrt(24)"


def test_gate_sw_vae_overfit_32_assets_fscore_99():
    """Phase 4 Acceptance Gate:
    Overfits 32 diverse procedural assets on RTX 5060 (< 5.2 GB training VRAM, 0 MB shared spill)
    achieving geometric reconstruction F-Score@0.01 >= 0.990!
    """
    # 1. Generate 32 diverse procedural assets across all 6 categories
    assets = generate_procedural_batch(count=32, num_workers=16)
    assert len(assets) == 32

    # 2. Train VAE inside strict training VRAM guard (< 5200 MB, 0 MB shared spill)
    with VRAMBudgetGuard(max_mb=5200) as guard:
        model, final_fscore, mean_norm, norm_std = train_vae_on_assets(
            assets=assets,
            num_epochs=120,
            lr=3e-3,
            max_tokens=256,
            device=torch.device(DEV),
        )

    target_norm = math.sqrt(LATENT_DIM)
    norm_err = abs(mean_norm - target_norm) / target_norm

    print(f"\n[Phase 4 Gate] SW-VAE 32-asset overfit:")
    print(f"  * Reconstruction F-Score@0.01: {final_fscore:.4f} (SLA: >= 0.990)")
    print(f"  * Latent shell radius: {mean_norm:.4f} vs sqrt(24)={target_norm:.4f} (err: {norm_err:.2%})")
    print(f"  * Peak training VRAM: {guard.peak_mb:.1f} MB (Cap: 5200 MB)")
    print(f"  * Shared GPU memory spill: {guard.shared_spill_mb:.1f} MB (Cap: 0 MB)")

    assert final_fscore >= 0.990, f"F-Score@0.01 {final_fscore:.4f} < 0.990 threshold!"
    assert norm_err < 0.05, f"Latent norm {mean_norm:.4f} deviates > 5% from sqrt(24)"
    assert guard.peak_mb <= 5200.0, f"Training VRAM {guard.peak_mb:.1f} MB exceeded 5.2 GB"
    assert guard.shared_spill_mb == 0.0, f"Shared GPU spill {guard.shared_spill_mb} MB > 0"
