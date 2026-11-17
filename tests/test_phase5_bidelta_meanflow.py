"""Phase 5 Acceptance Gate: BiDelta-SSA Backbone & OT-MeanFlow-3D (roadfinal.md §7 Phase 5).

Contract:
  1. Sinkhorn-Knopp OT pairing reduces mini-batch transport cost.
  2. BiDelta-SSA linear O(L) memory scaling verified up to L=4096 on RTX 5060.
  3. JVP-Free Interval-Splitting MeanFlow converges so 1-NFE / 2-NFE sampling
     matches 32-step ODE (< 4% relative latent error).
  4. Sym-Lock guarantees |x + mirror(x)| < 1e-5.
  5. NativeZeroConditioner + Stage1CoarseFlow run end-to-end inside VRAM guard.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from redis3d.core.hardware_guard import VRAMBudgetGuard
from redis3d.math.optimal_transport import ot_pair_batch, sinkhorn_knopp_ot
from redis3d.models.bidelta_ssa import BiDeltaSSABackbone
from redis3d.models.coarse_flow import Stage1CoarseFlow
from redis3d.models.conditioner import NativeZeroConditioner
from redis3d.models.latent_meanflow import (
    Stage2LatentMeanFlow,
    apply_sym_lock,
    verify_sym_lock_error,
)
from redis3d.models.sw_vae import LATENT_DIM
from redis3d.training.cpu_adamw import CPUAdamW
from redis3d.training.train_coarse import train_coarse_flow_step
from redis3d.training.train_meanflow import train_meanflow_on_latents

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real GPU required")
DEV = "cuda"


def test_sinkhorn_ot_reduces_transport_cost():
    """Verify Sinkhorn-Knopp OT produces doubly-stochastic plan and reduces cost."""
    torch.manual_seed(42)
    b, l, d = 16, 32, 24
    z1 = torch.randn(b, l, d, device=DEV) * 0.5 + 2.0
    z0 = torch.randn(b, l, d, device=DEV)

    raw_cost = (z0 - z1).pow(2).mean().item()
    z0_paired, perm, pi = ot_pair_batch(z0, z1, reg=0.05, num_iters=15)
    ot_cost = (z0_paired - z1).pow(2).mean().item()

    # Doubly stochastic check: rows and cols sum to 1/B
    row_sums = pi.sum(dim=1)
    col_sums = pi.sum(dim=0)
    expected = torch.full_like(row_sums, 1.0 / b)
    assert torch.allclose(row_sums, expected, atol=1e-2)
    assert torch.allclose(col_sums, expected, atol=1e-2)
    assert len(torch.unique(perm)) == b
    assert ot_cost <= raw_cost
    print(f"\n[Sinkhorn-OT] Transport MSE: raw={raw_cost:.4f} -> OT={ot_cost:.4f}")


def test_bidelta_ssa_linear_memory_scaling_up_to_4096():
    """Verify O(L) memory scaling of BiDelta-SSA from L=512 to L=4096 on RTX 5060."""
    torch.manual_seed(0)
    model = BiDeltaSSABackbone(
        dim=192,
        cond_dim=192,
        num_groups=2,
        num_heads=6,
        window_size=64,
        num_registers=16,
    ).to(DEV).eval()

    cond = torch.randn(1, 192, device=DEV)
    peaks_mb: dict[int, float] = {}

    with VRAMBudgetGuard(max_mb=2048):
        for seq_len in (512, 1024, 2048, 4096):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(DEV)
            base_mem = torch.cuda.memory_allocated(DEV)

            x = torch.randn(1, seq_len, 192, device=DEV)
            with torch.no_grad():
                out = model(x, cond)
            torch.cuda.synchronize()
            assert out.shape == (1, seq_len, 192)

            peak_act_mb = (torch.cuda.max_memory_allocated(DEV) - base_mem) / (1024.0 * 1024.0)
            peaks_mb[seq_len] = max(peak_act_mb, 0.1)

    ratio_4x = peaks_mb[4096] / max(peaks_mb[1024], 1e-3)
    print(f"\n[BiDelta-SSA Memory Scaling] L=512..4096 delta MB: {peaks_mb} (4096/1024 ratio={ratio_4x:.2f}x)")
    # Quadratic O(L^2) would be ~16x; linear O(L) should be <= 6.5x
    assert ratio_4x < 6.5, f"Non-linear memory growth detected: {ratio_4x:.2f}x"


def test_jvp_free_meanflow_1nfe_2nfe_matches_32step_ode():
    """Verify JVP-Free Interval-Splitting MeanFlow converges so 1-NFE/2-NFE matches 32-step ODE (< 4% error)."""
    torch.manual_seed(7)
    n_samples, seq_len, cond_dim = 32, 32, 128

    # Construct structured target distribution on hyperspherical shell conditioned on c
    cond_data = torch.randn(n_samples, cond_dim, device=DEV)
    proj = torch.randn(cond_dim, seq_len * LATENT_DIM, device=DEV) / math.sqrt(cond_dim)
    raw_z1 = (cond_data @ proj).view(n_samples, seq_len, LATENT_DIM)
    z1_data = raw_z1 / torch.sqrt(raw_z1.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    model = Stage2LatentMeanFlow(
        latent_dim=LATENT_DIM,
        model_dim=128,
        cond_dim=cond_dim,
        num_groups=1,
        num_heads=4,
        window_size=16,
        num_registers=8,
    ).to(DEV)

    model, ema_model, losses = train_meanflow_on_latents(
        model=model,
        z1_dataset=z1_data,
        cond_pooled_dataset=cond_data,
        num_steps=160,
        batch_size=16,
        lr=1e-3,
        max_vram_mb=4096,
        use_ot=True,
    )
    assert losses[-1] < losses[0], f"MeanFlow loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    # Compare 1-NFE and 2-NFE against 32-step Euler ODE from identical noise z0
    ema_model.eval()
    z0_eval = torch.randn(8, seq_len, LATENT_DIM, device=DEV)
    c_eval = cond_data[:8]

    z_ode = ema_model.sample_ode(z0_eval, c_eval, w_cfg=1.0, steps=32)
    z_1nfe = ema_model.sample_1nfe(z0_eval, c_eval, w_cfg=1.0)
    z_2nfe = ema_model.sample_2nfe(z0_eval, c_eval, w_cfg=1.0)

    ode_norm = torch.norm(z_ode, dim=-1).mean().clamp(min=1e-6)
    err_1nfe = (torch.norm(z_1nfe - z_ode, dim=-1).mean() / ode_norm).item()
    err_2nfe = (torch.norm(z_2nfe - z_ode, dim=-1).mean() / ode_norm).item()

    print(
        f"\n[OT-MeanFlow-3D] Loss {losses[0]:.4f}->{losses[-1]:.4f} | "
        f"1-NFE vs ODE32 err={err_1nfe*100:.2f}% | 2-NFE vs ODE32 err={err_2nfe*100:.2f}% (SLA < 4.0%)"
    )
    assert err_1nfe < 0.04, f"1-NFE error {err_1nfe*100:.2f}% exceeds 4% SLA"
    assert err_2nfe < 0.04, f"2-NFE error {err_2nfe*100:.2f}% exceeds 4% SLA"


def test_sym_lock_bilateral_symmetry_guarantee():
    """Verify Sym-Lock enforces |x + mirror(x)| < 1e-5 on mirrored geometry."""
    torch.manual_seed(123)
    half = torch.randn(128, 3, device=DEV) * 0.8
    half[:, 0] = half[:, 0].abs() + 0.05
    mirrored_half = half.clone()
    mirrored_half[:, 0] = -mirrored_half[:, 0]
    # Inject small asymmetry perturbation
    pts = torch.cat([half, mirrored_half + torch.randn_like(half) * 0.01], dim=0)

    locked = apply_sym_lock(pts, sym_axis=0)
    err = verify_sym_lock_error(locked, sym_axis=0)
    print(f"\n[Sym-Lock] Max |x + mirror(x)| = {err:.2e} (SLA < 1e-5)")
    assert err < 1e-5, f"Sym-Lock error {err:.2e} exceeds 1e-5"


def test_conditioner_and_stage1_coarse_flow_gate():
    """Verify NativeZeroConditioner + Stage1CoarseFlow forward and training step."""
    with VRAMBudgetGuard(max_mb=2048):
        cond_net = NativeZeroConditioner(cond_dim=128, image_size=64, patch_size=16, num_layers=2, num_heads=4).to(DEV)
        stage1 = Stage1CoarseFlow(in_channels=8, model_dim=128, cond_dim=128, num_groups=1, num_heads=4, window_size=16).to(DEV)
        ema_stage1 = copy.deepcopy(stage1).eval()
        opt = CPUAdamW(list(cond_net.parameters()) + list(stage1.parameters()), lr=1e-3)

        imgs = torch.rand(4, 3, 64, 64, device=DEV)
        prompts = ["beveled titanium gear", "armored mech visor", "octagonal flange", "low-poly crystal"]
        c_tokens, c_pooled = cond_net(images=imgs, prompts=prompts)
        assert c_pooled.shape == (4, 128)

        z1 = torch.randn(4, 32, 8, device=DEV)
        metrics = train_coarse_flow_step(stage1, ema_stage1, opt, z1, c_pooled, cond_tokens=c_tokens)
        assert math.isfinite(metrics["loss"])

        sample = stage1.sample_1nfe(torch.randn_like(z1), c_pooled, cond_tokens=c_tokens)
        assert sample.shape == z1.shape
        print(f"\n[Stage1 + Conditioner] Loss={metrics['loss']:.4f}, sample={tuple(sample.shape)}")
