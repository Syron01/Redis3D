"""Consilience-3D Best-of-4 Verifier & Physical Scorer (roadfinal.md §3.4, §7 Phase 7).

Evaluates K=4 parallel 1-NFE candidates decoded into sparse dual-grid clouds:
  S(k) = a1 * ConsensusChamfer^{-1}(k) + a2 * SilhouetteIoU(k, mask_ref)
       + a3 * EdgeCosine(k, I_edges) + a4 * TopoHealth(k)

Rejects single-pass hallucinations / extra limbs and locks onto the candidate
with strongest multi-candidate geometric consensus and pixel alignment.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F


class ConsilienceResult(NamedTuple):
    best_idx: int
    scores: torch.Tensor              # [K]
    consensus_chamfer: torch.Tensor   # [K]
    silhouette_iou: torch.Tensor      # [K]
    edge_cosine: torch.Tensor         # [K]
    topo_health: torch.Tensor         # [K]


def generate_sobol_latents(
    num_candidates: int,
    seq_len: int,
    latent_dim: int,
    device: torch.device,
    seed: int = 42,
) -> torch.Tensor:
    """Generate K quasi-orthogonal / Sobol-seeded Gaussian noise latents [K, L, D]."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sobol = torch.quasirandom.SobolEngine(dimension=latent_dim, scramble=True, seed=seed)
    # Draw K * L quasi-uniform points in (0, 1) and map via inverse normal CDF
    u = sobol.draw(num_candidates * seq_len).clamp(1e-5, 1.0 - 1e-5)
    z = torch.erfinv(2.0 * u - 1.0) * (2.0 ** 0.5)
    return z.view(num_candidates, seq_len, latent_dim).to(device=device, dtype=torch.float32)


def soft_splat_silhouette(
    coords: torch.Tensor,
    res: int = 64,
    sigma: float = 1.6,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable orthographic XY soft-splat rasterizer -> [B, res, res] in [0, 1].

    Args:
        coords: [B, L, 3] in [-1, 1]^3.
        res: Raster grid resolution (H=W=res).
        sigma: Gaussian splat radius in pixels.
        weights: Optional [B, L] per-point weights (e.g. edge/rank strength).
    """
    bsz, n_pts, _ = coords.shape
    device = coords.device

    # Map x, y in [-1, 1] to pixel coords [0, res-1] (y flipped for image convention)
    px = (coords[..., 0] + 1.0) * 0.5 * (res - 1)
    py = (1.0 - coords[..., 1]) * 0.5 * (res - 1)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(res, device=device, dtype=coords.dtype),
        torch.arange(res, device=device, dtype=coords.dtype),
        indexing="ij",
    )
    # [1, 1, res, res]
    gx = grid_x.view(1, 1, res, res)
    gy = grid_y.view(1, 1, res, res)

    # Chunk over points (chunk=512 uses ~8MB at res=64)
    acc = torch.zeros((bsz, res, res), device=device, dtype=coords.dtype)
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    chunk = 512
    for s in range(0, n_pts, chunk):
        e = min(s + chunk, n_pts)
        dx = gx - px[:, s:e].view(bsz, e - s, 1, 1)
        dy = gy - py[:, s:e].view(bsz, e - s, 1, 1)
        g = torch.exp(-(dx * dx + dy * dy) * inv_2s2)
        if weights is not None:
            g = g * weights[:, s:e].view(bsz, e - s, 1, 1)
        acc = acc + g.sum(dim=1)

    # Smooth saturation into [0, 1]
    return 1.0 - torch.exp(-acc * 0.85)


def compute_pairwise_chamfer(coords: torch.Tensor) -> torch.Tensor:
    """Compute symmetric Chamfer distance matrix [K, K] among K point clouds [K, L, 3]."""
    k = coords.shape[0]
    chamfer = torch.zeros((k, k), device=coords.device, dtype=coords.dtype)
    for i in range(k):
        for j in range(i + 1, k):
            d_ij = torch.cdist(coords[i], coords[j], p=2.0).pow(2)
            c = d_ij.min(dim=1).values.mean() + d_ij.min(dim=0).values.mean()
            chamfer[i, j] = c
            chamfer[j, i] = c
    return chamfer


def compute_topo_health(coords: torch.Tensor) -> torch.Tensor:
    """Score geometric coherence [K] in (0, 1]: penalizes collapsed or scattered clouds."""
    # Nearest-neighbor spacing uniformity + bounding box non-degeneracy
    k = coords.shape[0]
    scores = torch.zeros(k, device=coords.device, dtype=coords.dtype)
    for i in range(k):
        pts = coords[i]
        dists = torch.cdist(pts, pts, p=2.0)
        dists.fill_diagonal_(1e6)
        nn = dists.min(dim=-1).values
        # Outlier ratio: points > 4x median NN distance
        med = nn.median().clamp(min=1e-4)
        outlier_frac = (nn > 4.0 * med).float().mean()
        bbox_span = (pts.max(dim=0).values - pts.min(dim=0).values).min()
        non_collapsed = torch.sigmoid((bbox_span - 0.15) * 10.0)
        scores[i] = (1.0 - outlier_frac).clamp(0.0, 1.0) * non_collapsed
    return scores


class Consilience3DVerifier:
    """Best-of-K Consensus & Physical Alignment Scorer."""

    def __init__(
        self,
        alpha_consensus: float = 0.35,
        alpha_sil: float = 0.35,
        alpha_edge: float = 0.15,
        alpha_topo: float = 0.15,
        raster_res: int = 64,
    ):
        self.alpha_consensus = alpha_consensus
        self.alpha_sil = alpha_sil
        self.alpha_edge = alpha_edge
        self.alpha_topo = alpha_topo
        self.raster_res = raster_res

    @torch.no_grad()
    def score_and_select(
        self,
        coords: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_edges: Optional[torch.Tensor] = None,
        rank_logits: Optional[torch.Tensor] = None,
    ) -> ConsilienceResult:
        """Rank K candidates [K, L, 3] and return ConsilienceResult."""
        k = coords.shape[0]
        if ref_mask.dim() == 3:
            ref_mask = ref_mask[0]
        if ref_mask.shape[-2:] != (self.raster_res, self.raster_res):
            ref_m = F.interpolate(
                ref_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(self.raster_res, self.raster_res),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        else:
            ref_m = ref_mask.float()

        # 1. Consensus Chamfer^{-1}
        cd_mat = compute_pairwise_chamfer(coords)
        if k > 1:
            mean_cd = cd_mat.sum(dim=1) / float(k - 1)
        else:
            mean_cd = torch.zeros(1, device=coords.device)
        consensus_score = 1.0 / (1.0 + mean_cd * 10.0)

        # 2. Silhouette IoU
        pred_sil = soft_splat_silhouette(coords, res=self.raster_res, sigma=1.6)
        inter = (pred_sil * ref_m.unsqueeze(0)).sum(dim=(1, 2))
        union = (pred_sil + ref_m.unsqueeze(0) - pred_sil * ref_m.unsqueeze(0)).sum(dim=(1, 2)).clamp(min=1e-6)
        sil_iou = inter / union

        # 3. Edge Cosine
        if ref_edges is not None:
            if ref_edges.dim() == 3:
                ref_edges = ref_edges[0]
            if ref_edges.shape[-2:] != (self.raster_res, self.raster_res):
                ref_e = F.interpolate(
                    ref_edges.unsqueeze(0).unsqueeze(0).float(),
                    size=(self.raster_res, self.raster_res),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            else:
                ref_e = ref_edges.float()

            if rank_logits is not None:
                # Weight sharp Rank-2/3 edge/corner vertices higher
                probs = F.softmax(rank_logits, dim=-1)
                edge_w = (probs[..., 2] + probs[..., 3]).clamp(0.1, 1.0)
            else:
                edge_w = None
            pred_e = soft_splat_silhouette(coords, res=self.raster_res, sigma=1.2, weights=edge_w)
            edge_cos = F.cosine_similarity(
                pred_e.view(k, -1), ref_e.view(1, -1).expand(k, -1), dim=1
            ).clamp(0.0, 1.0)
        else:
            edge_cos = torch.ones(k, device=coords.device)

        # 4. TopoHealth
        topo = compute_topo_health(coords)

        total = (
            self.alpha_consensus * consensus_score
            + self.alpha_sil * sil_iou
            + self.alpha_edge * edge_cos
            + self.alpha_topo * topo
        )
        best_idx = int(torch.argmax(total).item())
        return ConsilienceResult(
            best_idx=best_idx,
            scores=total,
            consensus_chamfer=mean_cd,
            silhouette_iou=sil_iou,
            edge_cosine=edge_cos,
            topo_health=topo,
        )
