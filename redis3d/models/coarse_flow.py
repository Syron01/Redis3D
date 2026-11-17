"""Stage 1: Depth-6 Coarse Occupancy OT-MeanFlow (roadfinal.md §3.3, §4.1, §7 Phase 5).

Generates coarse sparse occupancy / latent structure at octree depth 6 (64^3)
conditioned on NativeZeroConditioner outputs, using Dense BiDelta-SSA + OT-MeanFlow.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from redis3d.models.bidelta_ssa import BiDeltaSSABackbone


def sinusoidal_time_embed(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """Sinusoidal embedding for scalar flow time t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb.to(t.dtype)


class Stage1CoarseFlow(nn.Module):
    """Depth-6 Coarse Occupancy OT-MeanFlow."""

    def __init__(
        self,
        in_channels: int = 8,
        model_dim: int = 192,
        cond_dim: int = 256,
        num_groups: int = 2,
        num_heads: int = 6,
        window_size: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_dim = model_dim
        self.cond_dim = cond_dim

        self.in_proj = nn.Linear(in_channels, model_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(192, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.backbone = BiDeltaSSABackbone(
            dim=model_dim,
            cond_dim=cond_dim,
            num_groups=num_groups,
            num_heads=num_heads,
            window_size=window_size,
            num_registers=16,
        )
        self.out_proj = nn.Linear(model_dim, in_channels)

    def forward(
        self,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        cond_pooled: torch.Tensor,
        w_cfg: Optional[torch.Tensor] = None,
        cond_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict average velocity u_theta(z_t, r, t, c, w) -> [B, L, C]."""
        bsz = z_t.shape[0]
        if w_cfg is None:
            w_cfg = torch.ones(bsz, device=z_t.device, dtype=z_t.dtype)

        t_emb = torch.cat(
            [
                sinusoidal_time_embed(r, 64),
                sinusoidal_time_embed(t, 64),
                sinusoidal_time_embed(w_cfg, 64),
            ],
            dim=-1,
        )
        cond_vec = cond_pooled + self.time_mlp(t_emb)
        h = self.in_proj(z_t)
        h = self.backbone(h, cond_vec, cond_tokens=cond_tokens)
        return self.out_proj(h)

    @torch.no_grad()
    def sample_1nfe(
        self,
        z0: torch.Tensor,
        cond_pooled: torch.Tensor,
        w_cfg: float = 1.5,
        cond_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = z0.shape[0]
        r = torch.zeros(bsz, device=z0.device, dtype=z0.dtype)
        t = torch.ones(bsz, device=z0.device, dtype=z0.dtype)
        w = torch.full((bsz,), w_cfg, device=z0.device, dtype=z0.dtype)
        return z0 + self.forward(z0, r, t, cond_pooled, w_cfg=w, cond_tokens=cond_tokens)
