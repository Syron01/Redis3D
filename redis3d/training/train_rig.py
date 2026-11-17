"""Training loop for Axion-Rig Skeleton Tree Net (roadfinal.md §7 Phase 8)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from redis3d.postprocess.auto_rig import SkeletonTreeNet
from redis3d.training.cpu_adamw import CPUAdamW


def train_rig_step(
    model: SkeletonTreeNet,
    optimizer: CPUAdamW,
    pts: torch.Tensor,
    gt_joints: torch.Tensor,
) -> float:
    """Single supervised joint prediction step."""
    model.train()
    pred = model(pts)
    loss = F.smooth_l1_loss(pred, gt_joints)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
