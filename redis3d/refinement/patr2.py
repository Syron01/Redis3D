"""PATR-2 Differentiable QEF & Intrinsic Texture Lock (roadfinal.md §3.4, §7 Phase 7).

Runs 6 Adam steps on the active dual vertices v_{c,s} [L, 3] and leaf PBR tokens
[L, 7] of winning candidate k* using differentiable soft-splat rasterization
against (I_albedo, I_normal, I_edges, mask_ref):
  - Clamps vertex displacements inside cell bounds (and domain [-1, 1]^3).
  - Preserves CAD-Snap rank constraints:
      * Rank 3 (Sharp Corners): pinned tightly to anchor (zero rounding).
      * Rank 2 (Sharp Creases): low displacement penalty along edge.
      * Rank 1 (Planar Sheets): free to align silhouette & normal to pixels.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

from redis3d.models.consilience_verifier import soft_splat_silhouette


class PATR2Result(NamedTuple):
    coords: torch.Tensor       # [L, 3] refined dual vertices
    pbr: torch.Tensor          # [L, 7] refined PBR tokens
    initial_iou: float
    final_iou: float


def _soft_splat_attribute(
    coords: torch.Tensor,
    attr: torch.Tensor,
    res: int = 64,
    sigma: float = 1.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable splat of per-point attributes [1, L, C] -> ([1, C, res, res], [1, res, res])."""
    bsz, n_pts, channels = attr.shape
    device = coords.device

    px = (coords[..., 0] + 1.0) * 0.5 * (res - 1)
    py = (1.0 - coords[..., 1]) * 0.5 * (res - 1)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(res, device=device, dtype=coords.dtype),
        torch.arange(res, device=device, dtype=coords.dtype),
        indexing="ij",
    )
    gx = grid_x.view(1, 1, res, res)
    gy = grid_y.view(1, 1, res, res)

    w_acc = torch.zeros((bsz, res, res), device=device, dtype=coords.dtype)
    c_acc = torch.zeros((bsz, channels, res, res), device=device, dtype=coords.dtype)
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)

    chunk = 512
    for s in range(0, n_pts, chunk):
        e = min(s + chunk, n_pts)
        dx = gx - px[:, s:e].view(bsz, e - s, 1, 1)
        dy = gy - py[:, s:e].view(bsz, e - s, 1, 1)
        g = torch.exp(-(dx * dx + dy * dy) * inv_2s2)  # [B, chunk, H, W]
        w_acc = w_acc + g.sum(dim=1)
        # [B, chunk, C] x [B, chunk, H, W] -> [B, C, H, W]
        a_chunk = attr[:, s:e, :].transpose(1, 2)  # [B, C, chunk]
        c_acc = c_acc + torch.einsum("bck,bkhw->bchw", a_chunk, g)

    alpha = 1.0 - torch.exp(-w_acc * 0.85)
    img = c_acc / w_acc.unsqueeze(1).clamp(min=1e-4)
    return img, alpha


class PATR2Refiner:
    """6-step differentiable QEF & PBR pixel-lock optimizer."""

    def __init__(
        self,
        num_steps: int = 6,
        lr_coords: float = 0.04,
        lr_pbr: float = 0.08,
        raster_res: int = 64,
        max_cell_disp: float = 0.12,
    ):
        self.num_steps = num_steps
        self.lr_coords = lr_coords
        self.lr_pbr = lr_pbr
        self.raster_res = raster_res
        self.max_cell_disp = max_cell_disp

    def refine(
        self,
        coords: torch.Tensor,
        pbr: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_albedo: Optional[torch.Tensor] = None,
        ref_edges: Optional[torch.Tensor] = None,
        ranks: Optional[torch.Tensor] = None,
    ) -> PATR2Result:
        """Optimize dual vertices [L, 3] and PBR [L, 7] against pixel references.

        Args:
            coords: [L, 3] initial dual vertices in [-1, 1]^3.
            pbr: [L, 7] initial PBR tokens in [0, 1].
            ref_mask: [H, W] or [1, H, W] target binary/soft silhouette in [0, 1].
            ref_albedo: Optional [3, H, W] de-lit target albedo in [0, 1].
            ref_edges: Optional [H, W] or [1, H, W] target edge map in [0, 1].
            ranks: Optional [L] CAD-Snap rank in {1, 2, 3}.
        """
        c_init = coords.detach().clone()
        p_init = pbr.detach().clone()

        if ref_mask.dim() == 2:
            ref_m = ref_mask.unsqueeze(0).unsqueeze(0).float()
        elif ref_mask.dim() == 3:
            ref_m = ref_mask.unsqueeze(0).float() if ref_mask.shape[0] != 1 else ref_mask.unsqueeze(1).float()
        else:
            ref_m = ref_mask.float()
        ref_m = F.interpolate(ref_m, size=(self.raster_res, self.raster_res), mode="bilinear", align_corners=False).squeeze(1)

        if ref_albedo is not None:
            ra = ref_albedo.unsqueeze(0) if ref_albedo.dim() == 3 else ref_albedo
            ra = F.interpolate(ra.float(), size=(self.raster_res, self.raster_res), mode="bilinear", align_corners=False)
        else:
            ra = None

        if ref_edges is not None:
            re = ref_edges.unsqueeze(0).unsqueeze(0) if ref_edges.dim() == 2 else ref_edges.unsqueeze(0)
            re = F.interpolate(re.float(), size=(self.raster_res, self.raster_res), mode="bilinear", align_corners=False).squeeze(1)
        else:
            re = None

        # CAD-Snap rank stiffness: Rank 3 (corners) pinned, Rank 2 (edges) semi-stiff, Rank 1 (planar) free
        if ranks is not None:
            r_long = ranks.detach().long()
            stiffness = torch.where(r_long >= 3, 25.0, torch.where(r_long == 2, 4.0, 0.2)).unsqueeze(-1)
        else:
            stiffness = torch.full((c_init.shape[0], 1), 0.2, device=c_init.device)

        with torch.no_grad():
            init_sil = soft_splat_silhouette(c_init.unsqueeze(0), res=self.raster_res, sigma=1.6)
            inter0 = (init_sil * ref_m).sum()
            union0 = (init_sil + ref_m - init_sil * ref_m).sum().clamp(min=1e-6)
            initial_iou = float((inter0 / union0).item())

        v_opt = c_init.clone().detach().requires_grad_(True)
        p_opt = p_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam(
            [
                {"params": [v_opt], "lr": self.lr_coords},
                {"params": [p_opt], "lr": self.lr_pbr},
            ]
        )

        for _ in range(self.num_steps):
            optimizer.zero_grad()
            v_b = v_opt.unsqueeze(0)

            if ra is not None:
                albedo_pred, sil_pred = _soft_splat_attribute(
                    v_b, torch.clamp(p_opt[:, :3], 0.0, 1.0).unsqueeze(0), res=self.raster_res, sigma=1.6
                )
                loss_albedo = F.l1_loss(albedo_pred * ref_m.unsqueeze(1), ra * ref_m.unsqueeze(1))
            else:
                sil_pred = soft_splat_silhouette(v_b, res=self.raster_res, sigma=1.6)
                loss_albedo = 0.0

            # Soft IoU + BCE silhouette lock
            inter = (sil_pred * ref_m).sum()
            union = (sil_pred + ref_m - sil_pred * ref_m).sum().clamp(min=1e-6)
            loss_sil = (1.0 - inter / union) + 0.25 * F.mse_loss(sil_pred, ref_m)

            loss_edge = 0.0
            if re is not None and ranks is not None:
                edge_w = (ranks >= 2).float().unsqueeze(0).clamp(min=0.1)
                e_pred = soft_splat_silhouette(v_b, res=self.raster_res, sigma=1.2, weights=edge_w)
                loss_edge = 0.3 * F.mse_loss(e_pred, re)

            # CAD-Snap rank-preserving anchor regularization
            loss_qef = (stiffness * (v_opt - c_init).pow(2)).mean()

            loss = loss_sil + 0.5 * loss_albedo + loss_edge + loss_qef
            loss.backward()
            optimizer.step()

            # Clamp inside cell-local bounds and [-1, 1]^3 domain
            with torch.no_grad():
                delta = (v_opt - c_init).clamp(-self.max_cell_disp, self.max_cell_disp)
                # Pin Rank-3 corners strictly
                if ranks is not None:
                    delta = torch.where((ranks >= 3).unsqueeze(-1), torch.zeros_like(delta), delta)
                v_opt.copy_((c_init + delta).clamp(-1.0, 1.0))
                p_opt.clamp_(0.0, 1.0)

        with torch.no_grad():
            final_sil = soft_splat_silhouette(v_opt.detach().unsqueeze(0), res=self.raster_res, sigma=1.6)
            inter1 = (final_sil * ref_m).sum()
            union1 = (final_sil + ref_m - final_sil * ref_m).sum().clamp(min=1e-6)
            final_iou = float((inter1 / union1).item())

        return PATR2Result(
            coords=v_opt.detach(),
            pbr=p_opt.detach(),
            initial_iou=initial_iou,
            final_iou=final_iou,
        )
