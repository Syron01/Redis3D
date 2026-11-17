"""Training loop for Stage 2 W-OVoxel^2 OT-MeanFlow-3D (roadfinal.md §7 Phase 5).

Uses:
  - CPUAdamW pinned DDR5 offload (0 MB GPU optimizer state).
  - Sinkhorn-Knopp OT mini-batch pairing.
  - JVP-Free Interval-Splitting MeanFlow loss with EMA target network.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch

from redis3d.core.hardware_guard import VRAMBudgetGuard
from redis3d.models.latent_meanflow import Stage2LatentMeanFlow, compute_meanflow_loss
from redis3d.training.cpu_adamw import CPUAdamW
from redis3d.training.train_coarse import update_ema


def train_meanflow_on_latents(
    model: Stage2LatentMeanFlow,
    z1_dataset: torch.Tensor,
    cond_pooled_dataset: torch.Tensor,
    num_steps: int = 120,
    batch_size: int = 8,
    lr: float = 2e-3,
    ema_decay: float = 0.98,
    max_vram_mb: int = 5200,
    use_ot: bool = True,
) -> tuple[Stage2LatentMeanFlow, Stage2LatentMeanFlow, list[float]]:
    """Train Stage2LatentMeanFlow on a latent tensor dataset with CPUAdamW.

    Args:
        model: Stage2LatentMeanFlow instance on target device.
        z1_dataset: [N, L, 24] target SW-VAE latents.
        cond_pooled_dataset: [N, cond_dim] conditioning vectors.
        num_steps: Training steps.
        batch_size: Mini-batch size.
        lr: Learning rate for CPUAdamW.
        ema_decay: EMA decay for interval-splitting target.
        max_vram_mb: Guardrail cap.
        use_ot: Enable Sinkhorn-OT pairing.

    Returns:
        (model, ema_model, loss_history)
    """
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = CPUAdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_samples = z1_dataset.shape[0]
    losses: list[float] = []

    with VRAMBudgetGuard(max_mb=max_vram_mb):
        for step in range(num_steps):
            # Cosine LR decay to eliminate constant-LR jitter near convergence
            lr_scale = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, num_steps)))
            for pg in optimizer.param_groups:
                pg["lr"] = lr * lr_scale

            model.train()
            idx = torch.randint(0, n_samples, (min(batch_size, n_samples),), device=z1_dataset.device)
            z1 = z1_dataset[idx]
            c_pool = cond_pooled_dataset[idx]
            z0 = torch.randn_like(z1)

            loss, metrics = compute_meanflow_loss(
                model=model,
                ema_model=ema_model,
                z0=z0,
                z1=z1,
                cond_pooled=c_pool,
                use_ot=use_ot,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema_model, model, decay=ema_decay)

            losses.append(metrics["loss"])

    return model, ema_model, losses
