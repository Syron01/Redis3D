"""Phase 1 acceptance gate — W-OVoxel^2 + CAD-Snap math core (roadfinal.md §7).

Shapes: sharp boolean CAD block, open wavy cloth sheet, double-walled hollow
shell, two touching semantic parts (+ near-touching dual-sheet spheres).
Pass: lossless Chamfer < 1e-5, sharp 90-deg normal cosine > 0.999;
pruned >= 5.0x token compression, F-Score@0.01 >= 0.995, extraction < 120 ms.
All heavy runs inside VRAMBudgetGuard(2048) on the real RTX 5060.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from redis3d.core.hardware_guard import VRAMBudgetGuard
from redis3d.math.dual_contouring import (
    HermiteData,
    SheetInfo,
    compute_sheets,
    dual_contour_mesh,
    leaf_tokens,
    sample_heightfield_hermite,
    sample_sdf_hermite,
)
from redis3d.math.morton import MortonIndex, morton_decode, morton_encode, token_key
from redis3d.math.octahedral import oct_decode, oct_encode
from redis3d.math.qef import cad_snap_rank, snap_corner_vertex, solve_qef
from redis3d.math.wavelet_octree import decode_to_fine, haar_lift_prune

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="real GPU required")
DEV = "cuda"


# ---------- analytic shapes (exact SDFs, clean-room) ----------

def box_sdf(p, lo, hi):
    lo = torch.tensor(lo, device=p.device, dtype=p.dtype)
    hi = torch.tensor(hi, device=p.device, dtype=p.dtype)
    c, r = (lo + hi) / 2, (hi - lo) / 2
    q = (p - c).abs() - r
    return q.clamp_min(0).norm(dim=-1) + q.amax(dim=-1).clamp_max(0)


def cad_sdf(p):
    return torch.minimum(
        box_sdf(p, [-0.65, -0.55, -0.45], [0.15, 0.55, 0.45]),
        box_sdf(p, [-0.15, -0.55, -0.45], [0.65, 0.25, 0.45]),
    )


def cloth_height(xy):
    x, y = xy.unbind(-1)
    return 0.05 * torch.sin(2 * x) * torch.cos(2 * y)


def cloth_grad(xy):
    x, y = xy.unbind(-1)
    return torch.stack([0.1 * torch.cos(2 * x) * torch.cos(2 * y),
                        -0.1 * torch.sin(2 * x) * torch.sin(2 * y)], dim=-1)


def shell_sdf(p):
    r = p.norm(dim=-1)
    return torch.maximum(r - 0.2, 0.16 - r)


def touch_sdf(p):
    return torch.minimum(
        box_sdf(p, [-0.6, -0.4, -0.4], [0.0, 0.4, 0.4]),
        box_sdf(p, [0.0, -0.4, -0.4], [0.6, 0.4, 0.4]),
    )


def touch_part(p):
    return (p[:, 0] >= 0).to(torch.int64)


def twin_sdf(p):
    ra = (p - torch.tensor([0.1513, 0.0, 0.0], device=p.device)).norm(dim=-1) - 0.15
    rb = (p - torch.tensor([-0.1493, 0.0, 0.0], device=p.device)).norm(dim=-1) - 0.15
    return torch.minimum(ra, rb)


def twin_part(p):
    return (p[:, 0] >= 0).to(torch.int64)


def sphere_sdf(p):
    return p.norm(dim=-1) - 0.35


# ---------- exact analytic normals (Hermite input contract) ----------

def box_grad(p, lo, hi):
    lo = torch.tensor(lo, device=p.device, dtype=p.dtype)
    hi = torch.tensor(hi, device=p.device, dtype=p.dtype)
    c, r = (lo + hi) / 2, (hi - lo) / 2
    q = (p - c).abs() - r
    outside = q.clamp_min(0)
    n_out = outside / outside.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ax = q.argmax(dim=-1)
    n_in = torch.nn.functional.one_hot(ax, 3).to(p.dtype) * torch.sign(p - c).gather(1, ax.unsqueeze(1))
    return torch.where((q > 0).any(dim=-1, keepdim=True), n_out, n_in)


def _union_grad(p, boxes):
    sdfs = torch.stack([box_sdf(p, lo, hi) for lo, hi in boxes], dim=-1)
    win = sdfs.argmin(dim=-1)
    grads = torch.stack([box_grad(p, lo, hi) for lo, hi in boxes], dim=1)
    return grads.gather(1, win.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)


_CAD_BOXES = ([[-0.65, -0.55, -0.45], [0.15, 0.55, 0.45]],
              [[-0.15, -0.55, -0.45], [0.65, 0.25, 0.45]])
_TOUCH_BOXES = ([[-0.6, -0.4, -0.4], [0.0, 0.4, 0.4]],
                [[0.0, -0.4, -0.4], [0.6, 0.4, 0.4]])


def cad_normal(p):
    return _union_grad(p, _CAD_BOXES)


def touch_normal(p):
    return _union_grad(p, _TOUCH_BOXES)


def shell_normal(p):
    r = p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    outward = p / r
    return torch.where((r - 0.2).abs() <= (0.16 - r).abs(), outward, -outward)


def twin_normal(p):
    ca = torch.tensor([0.1513, 0.0, 0.0], device=p.device)
    cb = torch.tensor([-0.1493, 0.0, 0.0], device=p.device)
    da, db = p - ca, p - cb
    na = da / da.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    nb = db / db.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where((da.norm(dim=-1) <= db.norm(dim=-1)).unsqueeze(-1), na, nb)


def sphere_normal(p):
    return p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)


# ---------- measurement helpers ----------

def sample_mesh_points(mesh, n=30000, seed=0):
    v, f = mesh.vertices, mesh.faces
    tris = v[f]
    area = torch.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0], dim=-1).norm(dim=-1)
    cdf = area.cumsum(0)
    g = torch.Generator(device=v.device).manual_seed(seed)
    u = torch.rand(n, device=v.device, generator=g) * cdf[-1]
    idx = torch.searchsorted(cdf, u).clamp(max=f.shape[0] - 1)
    r1 = torch.sqrt(torch.rand(n, device=v.device, generator=g))
    r2 = torch.rand(n, device=v.device, generator=g)
    a, b, c = 1 - r1, r1 * (1 - r2), r1 * r2
    return a.unsqueeze(-1) * tris[idx, 0] + b.unsqueeze(-1) * tris[idx, 1] + c.unsqueeze(-1) * tris[idx, 2]


def face_normals(mesh):
    tris = mesh.vertices[mesh.faces]
    n = torch.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0], dim=-1)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def fscore(points_a, points_b, tol=0.01):
    a = points_a.detach().cpu().numpy().astype(np.float64)
    b = points_b.detach().cpu().numpy().astype(np.float64)
    da, _ = cKDTree(b).query(a, k=1)
    db, _ = cKDTree(a).query(b, k=1)
    prec = float((da < tol).mean())
    rec = float((db < tol).mean())
    return 2 * prec * rec / max(prec + rec, 1e-12), prec, rec


def constant_pbr(hd, albedo=(0.75, 0.75, 0.78)):
    C = hd.cell_keys.numel()
    t = torch.tensor([*albedo, 0.5, 0.1, 1.0, 0.0], device=hd.cell_keys.device)
    return t.view(1, 1, 7).expand(C, 2, 7).contiguous()


# ---------- unit tests ----------

def test_morton_roundtrip_and_lut():
    g = torch.Generator(device=DEV).manual_seed(1)
    coords = torch.randint(0, 1024, (50000, 3), device=DEV, generator=g)
    levels = torch.randint(5, 11, (50000,), device=DEV, generator=g)
    keys = morton_encode(coords, levels)
    back, lev_back = morton_decode(keys)
    assert torch.equal(back, coords)
    assert torch.equal(lev_back, levels)
    # level 10 sets bit 63 by contract (l<<60); keys are int64 bit containers
    idx = MortonIndex(keys[:1000])
    q = idx.query(keys[:1000])
    assert torch.equal(q.sort().values, torch.arange(1000, device=DEV))
    miss = idx.query(morton_encode(torch.full((10, 3), 1023, device=DEV), 10))
    assert (miss == -1).all()


def test_octahedral_lossless():
    g = torch.Generator(device=DEV).manual_seed(2)
    n = torch.randn(100000, 3, device=DEV, generator=g)
    n = n / n.norm(dim=-1, keepdim=True)
    poles = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], device=DEV)
    n = torch.cat([n, poles])
    o = oct_encode(n)
    assert o.abs().max().item() <= 1.0 + 1e-6
    rec = oct_decode(o)
    cos = (rec * n).sum(-1)
    assert cos.min().item() > 1 - 1e-6, f"min cosine {cos.min().item()}"


def test_analytic_eig_and_solve_match_linalg():
    from redis3d.math.qef import solve3, sym_eig3_desc

    g = torch.Generator(device=DEV).manual_seed(3)
    X = torch.randn(512, 3, 3, device=DEV, generator=g)
    C = X @ X.transpose(1, 2) + 0.1 * torch.eye(3, device=DEV)
    eig_ref = torch.linalg.eigvalsh(C).flip(-1)
    eig_ana = sym_eig3_desc(C)
    assert (eig_ana - eig_ref).abs().max().item() < 1e-3
    b = torch.randn(512, 3, device=DEV, generator=g)
    v_ref = torch.linalg.solve(C, b.unsqueeze(-1)).squeeze(-1)
    v_ana = solve3(C, b)
    assert (v_ana - v_ref).abs().max().item() < 1e-3


def test_qef_plane_edge_corner():
    # plane z=0.3
    p = torch.rand(1, 32, 3, device=DEV)
    p[0, :, 2] = 0.3
    n = torch.zeros(1, 32, 3, device=DEV)
    n[0, :, 2] = 1.0
    v, _ = solve_qef(p, n)
    assert abs(v[0, 2].item() - 0.3) < 1e-5
    # transverse components must equal the mass point (fp32 adjugate would drift)
    assert (v[0, :2] - p[0, :, :2].mean(dim=0)).abs().max().item() < 1e-5
    rank, _ = cad_snap_rank(n)
    assert rank[0].item() == 1
    # 90-deg crease: planes x=0.2 and y=0.7
    p2 = torch.rand(1, 32, 3, device=DEV)
    n2 = torch.zeros(1, 32, 3, device=DEV)
    p2[0, :16, 0] = 0.2
    n2[0, :16, 0] = 1.0
    p2[0, 16:, 1] = 0.7
    n2[0, 16:, 1] = 1.0
    v2, _ = solve_qef(p2, n2)
    assert abs(v2[0, 0].item() - 0.2) < 1e-4 and abs(v2[0, 1].item() - 0.7) < 1e-4
    assert abs(v2[0, 2].item() - p2[0, :, 2].mean().item()) < 1e-4
    rank2, _ = cad_snap_rank(n2)
    assert rank2[0].item() == 2
    # corner: three orthogonal planes
    p3 = torch.rand(1, 30, 3, device=DEV)
    n3 = torch.zeros(1, 30, 3, device=DEV)
    for s, ax in enumerate(range(3)):
        p3[0, s * 10:(s + 1) * 10, ax] = [0.2, 0.7, 0.4][ax]
        n3[0, s * 10:(s + 1) * 10, ax] = 1.0
    rank3, _ = cad_snap_rank(n3)
    assert rank3[0].item() == 3
    vc = snap_corner_vertex(p3, n3)
    assert (vc[0] - torch.tensor([0.2, 0.7, 0.4], device=DEV)).abs().max().item() < 1e-5
    # tilted plane: b = sum n (n.p) must use the full dot product, not n_i p_i
    nt = torch.tensor([1.0, 1.0, 1.0], device=DEV) / 3.0 ** 0.5
    ct = torch.tensor([0.3, 0.4, 0.5], device=DEV)
    pt = torch.rand(1, 24, 3, device=DEV)
    pt = pt - ((pt - ct) @ nt).unsqueeze(-1) * nt  # project onto plane
    nt_b = nt.view(1, 1, 3).expand_as(pt)
    vt, _ = solve_qef(pt, nt_b)
    assert abs((vt[0] @ nt).item() - (ct @ nt).item()) < 1e-5
    lat = vt[0] - pt[0].mean(dim=0)
    assert (lat - (lat @ nt) * nt).norm().item() < 1e-5


# ---------- lossless gates ----------

def test_lossless_cad_block_sharp_90():
    with VRAMBudgetGuard(2048):
        hd = sample_sdf_hermite(cad_sdf, level=8, normal_fn=cad_normal, device=DEV)
        sheets = compute_sheets(hd)
        mesh = dual_contour_mesh(hd, sheets)
        pts = sample_mesh_points(mesh)
        chamfer = cad_sdf(pts).abs().mean().item()
        assert chamfer < 1e-5, f"CAD chamfer {chamfer}"
        fn = face_normals(mesh)
        cent = mesh.vertices[mesh.faces].mean(dim=1)
        # +x face of box B (x ~ 0.65) and +y face of box B (y ~ 0.25) meet at 90 deg;
        # restrict to face interiors so adjacent-face triangles are not counted
        fx = (cent[:, 0] > 0.649) & (cent[:, 1] > -0.54) & (cent[:, 1] < 0.2) & (cent[:, 2].abs() < 0.44)
        fy = (cent[:, 1] > 0.249) & (cent[:, 1] < 0.26) & (cent[:, 0] < 0.6) & (cent[:, 0] > 0.2) & (cent[:, 2].abs() < 0.44)
        assert fx.any() and fy.any()
        cos_x = (fn[fx] @ torch.tensor([1.0, 0.0, 0.0], device=DEV)).mean().item()
        cos_y = (fn[fy] @ torch.tensor([0.0, 1.0, 0.0], device=DEV)).mean().item()
        assert cos_x > 0.999, f"+x face cosine {cos_x}"
        assert cos_y > 0.999, f"+y face cosine {cos_y}"
        # vertices pinned on the sharp edge line x=0.65, y=0.25
        edge_dist = ((mesh.vertices[:, 0] - 0.65).abs() + (mesh.vertices[:, 1] - 0.25).abs())
        on_edge = (edge_dist < 1e-5) & (mesh.vertices[:, 2].abs() < 0.45)
        assert on_edge.sum().item() > 10, "sharp edge vertices missing"
        ranks = sheets.ranks[sheets.valid]
        assert (ranks == 2).any() and (ranks == 3).any(), "CAD-Snap must detect creases and corners"
        print(f"\nCAD lossless: chamfer={chamfer:.2e}, cos_x={cos_x:.6f}, cos_y={cos_y:.6f}, "
              f"cells={hd.cell_keys.numel()}, edge_verts={int(on_edge.sum())}")


def test_lossless_open_cloth():
    with VRAMBudgetGuard(2048):
        hd = sample_heightfield_hermite(cloth_height, cloth_grad, level=10, device=DEV)
        sheets = compute_sheets(hd)
        mesh = dual_contour_mesh(hd, sheets)
        assert hd.open_sheet.all()
        pts = sample_mesh_points(mesh)
        err = (pts[:, 2] - cloth_height(pts[:, :2])).abs().mean().item()
        assert err < 1e-5, f"cloth chamfer {err}"
        assert mesh.face_open.all()
        print(f"\nCloth lossless: chamfer={err:.2e}, cells={hd.cell_keys.numel()}, tris={mesh.faces.shape[0]}")


def test_lossless_hollow_shell():
    with VRAMBudgetGuard(2048):
        hd = sample_sdf_hermite(shell_sdf, level=10, normal_fn=shell_normal, device=DEV)
        sheets = compute_sheets(hd)
        mesh = dual_contour_mesh(hd, sheets)
        pts = sample_mesh_points(mesh)
        chamfer = shell_sdf(pts).abs().mean().item()
        assert chamfer < 1e-5, f"shell chamfer {chamfer}"
        r = pts.norm(dim=-1)
        outer = (r > 0.18).sum().item()
        inner = (r < 0.18).sum().item()
        assert outer > 1000 and inner > 1000, "both walls must be extracted"
        print(f"\nShell lossless: chamfer={chamfer:.2e}, cells={hd.cell_keys.numel()}, "
              f"outer/inner samples={outer}/{inner}")


def test_lossless_touching_parts():
    with VRAMBudgetGuard(2048):
        hd = sample_sdf_hermite(touch_sdf, level=8, normal_fn=touch_normal, part_fn=touch_part, device=DEV)
        sheets = compute_sheets(hd)
        mesh = dual_contour_mesh(hd, sheets)
        pts = sample_mesh_points(mesh)
        chamfer = touch_sdf(pts).abs().mean().item()
        assert chamfer < 1e-5, f"touch chamfer {chamfer}"
        p0 = (mesh.face_part == 0).sum().item()
        p1 = (mesh.face_part == 1).sum().item()
        assert p0 > 100 and p1 > 100, "both semantic parts must survive"
        print(f"\nTouching parts lossless: chamfer={chamfer:.2e}, part0/part1 faces={p0}/{p1}")


def test_dual_sheet_near_touching_spheres():
    with VRAMBudgetGuard(2048):
        hd = sample_sdf_hermite(twin_sdf, level=10, normal_fn=twin_normal, part_fn=twin_part, device=DEV)
        sheets = compute_sheets(hd)
        mesh = dual_contour_mesh(hd, sheets)
        pts = sample_mesh_points(mesh)
        chamfer = twin_sdf(pts).abs().mean().item()
        assert chamfer < 1e-5, f"twin chamfer {chamfer}"
        dual = (hd.sheet_count == 2).sum().item()
        assert dual > 50, f"dual-sheet cells expected in the gap, got {dual}"
        assert (mesh.face_part == 0).any() and (mesh.face_part == 1).any()
        print(f"\nTwin spheres: chamfer={chamfer:.2e}, dual-sheet cells={dual}, cells={hd.cell_keys.numel()}")


# ---------- pruned gates ----------

def test_pruned_sphere_5x_fscore_120ms():
    with VRAMBudgetGuard(2048):
        hd = sample_sdf_hermite(sphere_sdf, level=9, normal_fn=sphere_normal, device=DEV)
        sheets = compute_sheets(hd)
        keys, feats, covs = leaf_tokens(hd, sheets, constant_pbr(hd))
        n_tokens = keys.numel()

        # tau is the compressor operating point (roadfinal tau_geom(l)); the
        # F-Score gate below independently enforces reconstructed quality
        pruned = haar_lift_prune(keys, feats, covs, level=9, tau_geom=0.5, tau_pbr=0.5)
        assert pruned.ratio >= 5.0, f"compression {pruned.ratio:.2f}x < 5x"

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decoded = decode_to_fine(pruned)
        valid = sheets.valid.reshape(-1)
        verts_dec = sheets.verts.clone()
        verts_dec.reshape(-1, 3)[valid] = decoded[:, :3]
        sheets_dec = SheetInfo(verts=verts_dec, normals=sheets.normals,
                               centroids=sheets.centroids, ranks=sheets.ranks, valid=sheets.valid)
        mesh = dual_contour_mesh(hd, sheets_dec)
        torch.cuda.synchronize()
        extract_ms = (time.perf_counter() - t0) * 1000
        assert extract_ms < 120.0, f"pruned extraction {extract_ms:.1f} ms >= 120 ms"

        pts = sample_mesh_points(mesh, n=50000)
        g = torch.Generator(device=DEV).manual_seed(7)
        gt = torch.randn(100000, 3, device=DEV, generator=g)
        gt = gt / gt.norm(dim=-1, keepdim=True) * 0.35
        f, prec, rec = fscore(pts, gt, tol=0.01)
        assert f >= 0.995, f"F-Score@0.01 {f:.4f} < 0.995"
        print(f"\nPruned sphere: tokens {n_tokens}->{pruned.keys.numel()} ({pruned.ratio:.2f}x), "
              f"extract={extract_ms:.1f} ms, F={f:.5f} (P={prec:.5f} R={rec:.5f})")


def test_prune_never_removes_sharp_features():
    with VRAMBudgetGuard(2048):
        hd = sample_sdf_hermite(cad_sdf, level=8, normal_fn=cad_normal, device=DEV)
        sheets = compute_sheets(hd)
        keys, feats, covs = leaf_tokens(hd, sheets, constant_pbr(hd))
        sharp = (sheets.ranks >= 2) & sheets.valid
        C = hd.cell_keys.numel()
        sharp_keys = token_key(hd.cell_keys.unsqueeze(1).expand(C, 2), torch.arange(2, device=DEV).expand(C, 2)).reshape(-1)[sharp.reshape(-1)]
        pruned = haar_lift_prune(keys, feats, covs, level=8, tau_geom=0.5, tau_pbr=0.5)
        assert pruned.ratio > 1.0, "smooth CAD faces should compress"
        idx = MortonIndex(pruned.keys)
        assert (idx.query(sharp_keys) >= 0).all(), "rank-2/3 tokens were pruned!"
        print(f"\nCAD prune: ratio={pruned.ratio:.2f}x, sharp tokens preserved={sharp_keys.numel()}")