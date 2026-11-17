"""Training loop for Stage 1 Depth-6 Coarse Occupancy OT-MeanFlow (roadfinal.md §7 Phase 5)."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn.functional as F

from redis3d.math.optimal_transport import ot_pair_batch
from redis3d.models.coarse_flow import Stage1CoarseFlow
from redis3d.training.cpu_adamw import CPUAdamW


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.995) -> None:
    """Update EMA target parameters."""
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train_coarse_flow_step(
    model: Stage1CoarseFlow,
    ema_model: Stage1CoarseFlow,
    optimizer: CPUAdamW,
    z1: torch.Tensor,
    cond_pooled: torch.Tensor,
    cond_tokens: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Run one Sinkhorn-OT + Interval-Splitting MeanFlow step for Stage 1."""
    model.train()
    bsz = z1.shape[0]
    device = z1.device
    dtype = z1.dtype

    z0 = torch.randn_like(z1)
    if bsz > 1:
        z0, _, _ = ot_pair_batch(z0, z1, reg=0.05, num_iters=15)

    w_cfg = torch.ones(bsz, device=device, dtype=dtype)

    # Boundary loss (r = t)
    t_b = torch.rand(bsz, device=device, dtype=dtype)
    z_tb = (1.0 - t_b.view(bsz, 1, 1)) * z0 + t_b.view(bsz, 1, 1) * z1
    u_b = model(z_tb, t_b, t_b, cond_pooled, w_cfg=w_cfg, cond_tokens=cond_tokens)
    loss_b = F.mse_loss(u_b, z1 - z0)

    # Interval split loss
    r = torch.rand(bsz, device=device, dtype=dtype) * 0.5
    t = (r + 0.15 + 0.35 * torch.rand(bsz, device=device, dtype=dtype)).clamp(max=1.0)
    m = 0.5 * (r + t)
    z_r = (1.0 - r.view(bsz, 1, 1)) * z0 + r.view(bsz, 1, 1) * z1

    with torch.no_grad():
        u1 = ema_model(z_r, r, m, cond_pooled, w_cfg=w_cfg, cond_tokens=cond_tokens)
        zm = z_r + (m - r).view(bsz, 1, 1) * u1
        u2 = ema_model(zm, m, t, cond_pooled, w_cfg=w_cfg, cond_tokens=cond_tokens)
        target = 0.5 * (u1 + u2)

    u_pred = model(z_r, r, t, cond_pooled, w_cfg=w_cfg, cond_tokens=cond_tokens)
    loss_s = F.mse_loss(u_pred, target.detach())

    loss = loss_b + loss_s
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    update_ema(ema_model, model, decay=0.99)

    return {
        "loss": float(loss.item()),
        "loss_boundary": float(loss_b.item()),
        "loss_split": float(loss_s.item()),
    }
