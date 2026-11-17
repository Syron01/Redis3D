"""Lossless octahedral normal codec: S^2 <-> [-1,1]^2 (roadfinal.md §3.1)."""

from __future__ import annotations

import torch


def oct_encode(n: torch.Tensor) -> torch.Tensor:
    """Unit normals [..., 3] -> octahedral coords [..., 2] in [-1, 1]^2."""
    n = n / n.abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = n.unbind(-1)
    neg = z < 0
    sx = torch.where(x >= 0, 1.0, -1.0)
    sy = torch.where(y >= 0, 1.0, -1.0)
    ox = torch.where(neg, (1.0 - y.abs()) * sx, x)
    oy = torch.where(neg, (1.0 - x.abs()) * sy, y)
    return torch.stack([ox, oy], dim=-1)


def oct_decode(o: torch.Tensor) -> torch.Tensor:
    """Octahedral coords [..., 2] -> unit normals [..., 3]."""
    x, y = o.unbind(-1)
    z = 1.0 - x.abs() - y.abs()
    t = (-z).clamp_min(0.0)
    x = x + torch.where(x >= 0, -t, t)
    y = y + torch.where(y >= 0, -t, t)
    n = torch.stack([x, y, z], dim=-1)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)