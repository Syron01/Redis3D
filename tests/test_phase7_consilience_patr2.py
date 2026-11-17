"""Phase 7 Acceptance Gate: Consilience-3D & PATR-2 Refiner (roadfinal.md §7 Phase 7).

Contract:
  1. Consilience-3D rejects injected faulty candidates (hallucinated extra limb /
     shifted silhouette) and selects the clean consensus candidate.
  2. Consilience-3D + PATR-2 improves Silhouette IoU by >= +6% over single-seed pass
     on controlled perturbed/occluded inputs while preserving Rank-3 sharp corners.
  3. Combined Consilience-3D + PATR-2 executes in <= 700 ms on RTX 5060 and stays
     strictly inside <= 2.0 GB VRAM.
"""

from __future__ import annotations

import time

import pytest
import torch

from redis3d.core.hardware_guard import VRAMBudgetGuard
from redis3d.models.consilience_verifier import (
    Consilience3DVerifier,
    soft_splat_silhouette,
)
from redis3d.models.pipeline import Redis3DProPipeline
from redis3d.refinement.patr2 import PATR2Refiner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real GPU required")
DEV = "cuda"


def _make_cube_cloud(n_pts: int = 512, half: float = 0.55) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a clean cube surface point cloud [n_pts, 3] and CAD-Snap ranks [n_pts]."""
    torch.manual_seed(0)
    pts = (torch.rand(n_pts, 3, device=DEV) * 2.0 - 1.0) * half
    # Project each point onto one of the 6 cube faces
    face_idx = torch.randint(0, 6, (n_pts,), device=DEV)
    for f in range(6):
        mask = face_idx == f
        axis = f // 2
        sign = 1.0 if (f % 2 == 0) else -1.0
        pts[mask, axis] = sign * half

    # Assign Rank 3 to corners, Rank 2 to edges, Rank 1 to faces
    near_bound = (pts.abs() > (half - 0.08)).sum(dim=-1)
    ranks = near_bound.clamp(1, 3)
    return pts, ranks


def test_consilience_3d_rejects_faulty_candidate():
    """Verify Consilience-3D rejects a hallucinated/extra-limb candidate."""
    base_pts, ranks = _make_cube_cloud(n_pts=384, half=0.5)
    ref_mask = soft_splat_silhouette(base_pts.unsqueeze(0), res=64, sigma=1.6)[0]

    # Candidates 0, 1, 2: clean consensus with tiny noise
    c0 = base_pts + 0.01 * torch.randn_like(base_pts)
    c1 = base_pts + 0.01 * torch.randn_like(base_pts)
    c2 = base_pts + 0.01 * torch.randn_like(base_pts)

    # Candidate 3: faulty hallucination (shifted + extra detached limb cluster)
    c3 = base_pts.clone()
    c3[:96] += torch.tensor([0.75, 0.75, 0.0], device=DEV)

    candidates = torch.stack([c0, c1, c2, c3], dim=0)
    verifier = Consilience3DVerifier()
    res = verifier.score_and_select(candidates, ref_mask=ref_mask)

    print(
        f"\n[Consilience-3D] Scores={[round(s, 4) for s in res.scores.tolist()]} | "
        f"Best={res.best_idx} (Faulty=3)"
    )
    assert res.best_idx != 3, "Consilience-3D failed to reject faulty candidate #3!"
    assert res.scores[3] < res.scores[res.best_idx]


def test_consilience_patr2_improves_iou_over_6pct_within_700ms_and_2gb():
    """Verify Consilience-3D + PATR-2 improves IoU >= +6%, runs <= 700 ms, <= 2.0 GB VRAM."""
    gt_pts, ranks = _make_cube_cloud(n_pts=512, half=0.55)
    with torch.no_grad():
        ref_mask = soft_splat_silhouette(gt_pts.unsqueeze(0), res=64, sigma=1.6)[0]
        ref_albedo = torch.full((3, 64, 64), 0.7, device=DEV)
        edge_w = (ranks >= 2).float().unsqueeze(0).clamp(min=0.1)
        ref_edges = soft_splat_silhouette(gt_pts.unsqueeze(0), res=64, sigma=1.2, weights=edge_w)[0]

    # Candidate 0 (single-seed baseline): drifted/shrunk by 22%
    c0 = gt_pts * 0.78 + torch.tensor([0.08, -0.06, 0.0], device=DEV)
    # Candidates 1, 2, 3: closer consensus candidates with moderate drift
    c1 = gt_pts * 0.90 + torch.tensor([0.04, -0.03, 0.0], device=DEV)
    c2 = gt_pts * 0.91 + torch.tensor([0.03, -0.02, 0.0], device=DEV)
    c3 = gt_pts * 0.89 + torch.tensor([-0.03, 0.03, 0.0], device=DEV)
    candidates = torch.stack([c0, c1, c2, c3], dim=0)

    # Measure single-seed IoU (candidate 0)
    with torch.no_grad():
        s0 = soft_splat_silhouette(c0.unsqueeze(0), res=64, sigma=1.6)
        single_iou = ((s0 * ref_mask).sum() / (s0 + ref_mask - s0 * ref_mask).sum().clamp(min=1e-6)).item()

    verifier = Consilience3DVerifier()
    patr2 = PATR2Refiner(num_steps=6, lr_coords=0.04, lr_pbr=0.08, raster_res=64, max_cell_disp=0.15)
    pbr_init = torch.full((gt_pts.shape[0], 7), 0.4, device=DEV)

    with VRAMBudgetGuard(max_mb=2048):
        # Warmup CUDA kernels (Consilience + PATR-2)
        _ = verifier.score_and_select(candidates, ref_mask=ref_mask, ref_edges=ref_edges)
        _ = patr2.refine(
            coords=candidates[0],
            pbr=pbr_init,
            ref_mask=ref_mask,
            ref_albedo=ref_albedo,
            ref_edges=ref_edges,
            ranks=ranks,
        )
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        cons_res = verifier.score_and_select(candidates, ref_mask=ref_mask, ref_edges=ref_edges)
        k_star = cons_res.best_idx
        patr_res = patr2.refine(
            coords=candidates[k_star],
            pbr=pbr_init,
            ref_mask=ref_mask,
            ref_albedo=ref_albedo,
            ref_edges=ref_edges,
            ranks=ranks,
        )
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

    iou_gain = patr_res.final_iou - single_iou
    # Verify Rank-3 sharp corners stayed pinned
    rank3_mask = ranks >= 3
    if rank3_mask.any():
        corner_disp = (patr_res.coords[rank3_mask] - candidates[k_star][rank3_mask]).abs().max().item()
        assert corner_disp < 1e-6, f"Rank-3 corners moved by {corner_disp}"

    print(
        f"\n[Consilience + PATR-2] Single-seed IoU={single_iou:.4f} -> Final IoU={patr_res.final_iou:.4f} "
        f"(Gain=+{iou_gain*100:.2f}%, SLA >= +6.0%) | Latency={dt_ms:.1f} ms (SLA <= 700 ms)"
    )
    assert iou_gain >= 0.06, f"IoU gain +{iou_gain*100:.2f}% < +6.0% SLA!"
    assert dt_ms <= 700.0, f"Combined latency {dt_ms:.1f} ms > 700 ms SLA!"


def test_full_pipeline_resident_mode_smoke():
    """Verify Redis3DProPipeline runs end-to-end inside 2048 MB VRAM."""
    pipeline = Redis3DProPipeline(vram_mode="resident", max_vram_mb=2048, device=DEV)
    img = torch.rand(1, 3, 128, 128, device=DEV)
    out = pipeline.generate(
        image=img,
        prompt="precision armored mech helmet",
        num_candidates=4,
        nfe_steps=1,
        patr_steps=6,
        seq_len=128,
        sym_lock=True,
    )
    assert out.coords.shape == (128, 3)
    assert out.pbr.shape == (128, 7)
    assert torch.isfinite(out.coords).all()
