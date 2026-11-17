"""Tikhonov-regularized 3x3 QEF + CAD-Snap eigen-rank analysis (roadfinal.md §3.1.1).

A = sum_m n_m n_m^T + lambda*I
b = sum_m n_m (n_m^T p_m) + lambda*p_mean
v* = clamp_[0,1]^3 (A^-1 b)

CAD-Snap ranks from the UNregularized normal covariance:
  sigma2/sigma1 < eps_plane                -> rank 1 planar sheet (prunable)
  sigma3/sigma1 < eps_edge <= sigma2/sigma1 -> rank 2 sharp crease (lock l>=9)
  sigma3/sigma1 >= eps_edge                -> rank 3 corner (pin to l=10)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

RANK_PLANAR = 1
RANK_EDGE = 2
RANK_CORNER = 3
MIN_LEVEL_EDGE = 9
MIN_LEVEL_CORNER = 10


def sym_eig3_desc(C: torch.Tensor) -> torch.Tensor:
    """Closed-form eigenvalues of symmetric 3x3 [B,3,3], descending.

    Trigonometric method: exact, elementwise, no cusolver batch limits.
    """
    a00, a11, a22 = C[:, 0, 0], C[:, 1, 1], C[:, 2, 2]
    a01, a02, a12 = C[:, 0, 1], C[:, 0, 2], C[:, 1, 2]
    tr = a00 + a11 + a22
    q = tr / 3.0
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = torch.sqrt(p2 / 6.0 + 1e-30)
    b00, b11, b22 = (a00 - q) / p, (a11 - q) / p, (a22 - q) / p
    b01, b02, b12 = a01 / p, a02 / p, a12 / p
    det_b = (
        b00 * (b11 * b22 - b12 * b12)
        - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02)
    )
    r = (det_b / 2.0).clamp(-1.0, 1.0)
    phi = torch.acos(r) / 3.0
    e1 = q + 2.0 * p * torch.cos(phi)
    e3 = q + 2.0 * p * torch.cos(phi + 2.0943951023931953)  # 2*pi/3
    e2 = tr - e1 - e3
    iso = p2 < 1e-24
    e1 = torch.where(iso, q, e1)
    e2 = torch.where(iso, q, e2)
    e3 = torch.where(iso, q, e3)
    return torch.stack([e1, e2, e3], dim=-1)


def solve3(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact 3x3 solve via cofactor adjugate: A [B,3,3], b [B,3] -> v [B,3]."""
    a, bb, c = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
    d, e, f = A[:, 1, 0], A[:, 1, 1], A[:, 1, 2]
    g, h, i = A[:, 2, 0], A[:, 2, 1], A[:, 2, 2]
    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    c10 = -(bb * i - c * h)
    c11 = a * i - c * g
    c12 = -(a * h - bb * g)
    c20 = bb * f - c * e
    c21 = -(a * f - c * d)
    c22 = a * e - bb * d
    det = a * c00 + bb * c01 + c * c02
    det = torch.where(det.abs() < 1e-20, torch.where(det >= 0, 1e-20, -1e-20), det)
    v0 = (c00 * b[:, 0] + c10 * b[:, 1] + c20 * b[:, 2]) / det
    v1 = (c01 * b[:, 0] + c11 * b[:, 1] + c21 * b[:, 2]) / det
    v2 = (c02 * b[:, 0] + c12 * b[:, 1] + c22 * b[:, 2]) / det
    return torch.stack([v0, v1, v2], dim=-1)


def solve3_spd(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cholesky solve for SPD 3x3. The cofactor adjugate cancels ~9 digits at
    QEF condition numbers (~count/reg), which even fp64 cannot absorb; the
    Schur-complement recursion stays stable. Run in fp64."""
    a00, a11, a22 = A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]
    a01, a02, a12 = A[:, 0, 1], A[:, 0, 2], A[:, 1, 2]
    l00 = a00.clamp_min(1e-30).sqrt()
    l10 = a01 / l00
    l11 = (a11 - l10 * l10).clamp_min(1e-30).sqrt()
    l20 = a02 / l00
    l21 = (a12 - l20 * l10) / l11
    l22 = (a22 - l20 * l20 - l21 * l21).clamp_min(1e-30).sqrt()
    y0 = b[:, 0] / l00
    y1 = (b[:, 1] - l10 * y0) / l11
    y2 = (b[:, 2] - l20 * y0 - l21 * y1) / l22
    v2 = y2 / l22
    v1 = (y1 - l21 * v2) / l11
    v0 = (y0 - l10 * v1 - l20 * v2) / l00
    return torch.stack([v0, v1, v2], dim=-1)


def _masked_covariance(normals: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """normals [B, M, 3] -> sum n n^T [B, 3, 3]."""
    if mask is not None:
        normals = normals * mask.unsqueeze(-1)
    return torch.einsum("bmi,bmj->bij", normals, normals)


def solve_qef(
    points: torch.Tensor,
    normals: torch.Tensor,
    mask: torch.Tensor | None = None,
    reg: float = 1e-6,
    clamp_unit: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched QEF. points/normals [B, M, 3] (points in cell-local coords).

    Returns (vertices [B, 3], mean squared plane residual [B]).
    """
    if mask is None:
        mask = torch.ones(points.shape[:2], dtype=points.dtype, device=points.device)
    # rank-1 sheets give cond(A) ~ count/reg (~1e6); fp32 adjugate cancellation
    # destroys the transverse vertex components, so accumulate/solve in fp64
    dtype = points.dtype
    points = points.double()
    normals = normals.double()
    w = mask.to(torch.float64)
    cnt = w.sum(dim=1).clamp_min(1.0)
    p_mean = (points * w.unsqueeze(-1)).sum(dim=1) / cnt.unsqueeze(-1)

    A = _masked_covariance(normals, w)
    A = A + reg * torch.eye(3, dtype=A.dtype, device=A.device)
    n_dot_p = torch.einsum("bmi,bmi->bm", normals, points)
    b = torch.einsum("bmi,bm,bm->bi", normals, n_dot_p, w) + reg * p_mean
    v = solve3_spd(A, b)
    if clamp_unit:
        v = v.clamp(0.0, 1.0)
    d = torch.einsum("bmi,bmi->bm", normals, points) - torch.einsum("bmi,bi->bm", normals, v)
    residual = ((d * d) * w).sum(dim=1) / cnt
    return v.to(dtype), residual.to(dtype)


def cad_snap_rank(
    normals: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps_plane: float = 0.01,
    eps_edge: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (rank [B] in {0,1,2,3}, eigenvalues [B,3] descending)."""
    C = _masked_covariance(normals, mask)
    eig = sym_eig3_desc(C)  # sigma1 >= sigma2 >= sigma3
    s1 = eig[:, 0].clamp_min(1e-12)
    r21 = eig[:, 1] / s1
    r31 = eig[:, 2] / s1
    rank = torch.full(eig.shape[:1], RANK_CORNER, dtype=torch.int64, device=eig.device)
    rank = torch.where(r31 < eps_edge, torch.full_like(rank, RANK_EDGE), rank)
    rank = torch.where(r21 < eps_plane, torch.full_like(rank, RANK_PLANAR), rank)
    rank = torch.where(eig[:, 0] < 1e-9, torch.zeros_like(rank), rank)
    return rank, eig


def snap_corner_vertex(points: torch.Tensor, normals: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Exact 3-plane intersection for rank-3 corners: pick the 3 most
    independent Hermite planes and solve N v = d without regularization."""
    if mask is None:
        mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
    B, M, _ = points.shape
    n = normals * mask.unsqueeze(-1)
    # Greedy triple: first strongest normal, then max |cross|, then max |det|.
    strength = n.norm(dim=-1) * mask
    i0 = strength.argmax(dim=1)
    n0 = n[torch.arange(B, device=n.device), i0]
    cross1 = torch.cross(n0.unsqueeze(1).expand_as(n), n, dim=-1).norm(dim=-1) * mask
    i1 = cross1.argmax(dim=1)
    n1 = n[torch.arange(B, device=n.device), i1]
    det2 = torch.einsum("bi,bmi->bm", torch.cross(n0, n1, dim=-1), n) * mask
    i2 = det2.abs().argmax(dim=1)
    n2 = n[torch.arange(B, device=n.device), i2]
    N = torch.stack([n0, n1, n2], dim=1)
    idx = torch.stack([i0, i1, i2], dim=1)
    p_sel = points.gather(1, idx.unsqueeze(-1).expand(B, 3, 3))
    d = torch.einsum("bki,bki->bk", N, p_sel)
    v = solve3(N.double(), d.double()).to(points.dtype)
    return v.clamp(0.0, 1.0)


@dataclass
class CADSnapInfo:
    rank: torch.Tensor
    eigenvalues: torch.Tensor
    min_level: torch.Tensor

    @property
    def prunable(self) -> torch.Tensor:
        return self.rank == RANK_PLANAR


def cad_snap_analyze(
    normals: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps_plane: float = 0.01,
    eps_edge: float = 0.1,
) -> CADSnapInfo:
    rank, eig = cad_snap_rank(normals, mask, eps_plane, eps_edge)
    min_level = torch.ones_like(rank)
    min_level = torch.where(rank == RANK_EDGE, torch.full_like(rank, MIN_LEVEL_EDGE), min_level)
    min_level = torch.where(rank == RANK_CORNER, torch.full_like(rank, MIN_LEVEL_CORNER), min_level)
    return CADSnapInfo(rank=rank, eigenvalues=eig, min_level=min_level)