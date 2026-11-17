"""Field-free Dual-Grid + Dual-Sheet mesh extractor (roadfinal.md §3.1).

Hermite samples come from primal edge crossings (SDF sign changes or analytic
heightfield intersections) — no density field is stored. Each active leaf cell
solves one QEF per surface sheet (K <= 2); sheets are split when two spatially
separated sample clusters carry opposing normals (hollow shells, touching
parts). Quads connect the dual vertices of the 4 cells around each crossing
edge; open sheets (cloth) close their rim with edge-point triangles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from redis3d.math.morton import MortonIndex, edge_key, edge_split, morton_decode, morton_encode, token_key
from redis3d.math.octahedral import oct_encode
from redis3d.math.qef import cad_snap_analyze, snap_corner_vertex, solve_qef

# corner index = dx + 2*dy + 4*dz
_CORNERS = torch.tensor(
    [[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=torch.int64
)
# 12 edges as (lower corner idx, upper corner idx, axis)
_EDGES12 = []
for _a in range(3):
    for _i in range(8):
        if (_i >> _a) & 1 == 0:
            _EDGES12.append((_i, _i | (1 << _a), _a))
_EDGE_LO = torch.tensor([e[0] for e in _EDGES12], dtype=torch.int64)
_EDGE_HI = torch.tensor([e[1] for e in _EDGES12], dtype=torch.int64)
_EDGE_AX = torch.tensor([e[2] for e in _EDGES12], dtype=torch.int64)
_M = 12


def _eval_chunked(fn: Callable, pts: torch.Tensor, chunk: int = 1 << 20) -> torch.Tensor:
    outs = []
    for i in range(0, pts.shape[0], chunk):
        outs.append(fn(pts[i : i + chunk]))
    return torch.cat(outs) if outs else torch.empty(0, device=pts.device)


def _world(coords: torch.Tensor, h: float) -> torch.Tensor:
    return -1.0 + coords.to(torch.float32) * h


def _sdf_gradient(sdf_fn: Callable, pts: torch.Tensor, eps: float) -> torch.Tensor:
    g = []
    for a in range(3):
        d = torch.zeros_like(pts)
        d[:, a] = eps
        g.append((_eval_chunked(sdf_fn, pts + d) - _eval_chunked(sdf_fn, pts - d)) / (2 * eps))
    n = torch.stack(g, dim=-1)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass
class HermiteData:
    level: int
    cell_keys: torch.Tensor       # [C]
    cell_coords: torch.Tensor     # [C, 3]
    samples_p: torch.Tensor       # [C, 12, 3] cell-local
    samples_n: torch.Tensor       # [C, 12, 3]
    sample_mask: torch.Tensor     # [C, 12] bool
    sheet_id: torch.Tensor        # [C, 12] int64 (0/1)
    sheet_count: torch.Tensor     # [C]
    open_sheet: torch.Tensor      # [C] bool
    part_id: torch.Tensor         # [C]
    edge_lower: torch.Tensor      # [E, 3]
    edge_axis: torch.Tensor       # [E]
    edge_point: torch.Tensor      # [E, 3] world
    edge_normal: torch.Tensor     # [E, 3]
    edge_va_neg: torch.Tensor     # [E] bool
    edge_part: torch.Tensor       # [E]
    cell_edge_slot: torch.Tensor  # [C, 12] edge id or -1

    @property
    def h(self) -> float:
        return 2.0 / (1 << self.level)


def _lookup_sorted(keys_sorted: torch.Tensor, vals_sorted: torch.Tensor, query: torch.Tensor):
    pos = torch.searchsorted(keys_sorted, query)
    pos_c = pos.clamp(max=keys_sorted.numel() - 1)
    hit = keys_sorted[pos_c] == query
    return pos_c, hit


def sample_sdf_hermite(
    sdf_fn: Callable,
    level: int,
    normal_fn: Optional[Callable] = None,
    part_fn: Optional[Callable] = None,
    device: str = "cuda",
    start_level: int = 5,
) -> HermiteData:
    """Coarse-to-fine active-cell traversal; only surface cells are refined."""
    dev = torch.device(device)
    corners = _CORNERS.to(dev)
    elo, ehi, eax = _EDGE_LO.to(dev), _EDGE_HI.to(dev), _EDGE_AX.to(dev)

    L = start_level
    n = 1 << L
    g = torch.arange(n + 1, device=dev, dtype=torch.int64)
    verts = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1).reshape(-1, 3)
    vals = _eval_chunked(sdf_fn, _world(verts, 2.0 / n))
    vk = morton_encode(verts, L)
    order = torch.argsort(vk)
    vk_s, vals_s = vk[order], vals[order]

    cell_coords = torch.stack(
        torch.meshgrid(
            torch.arange(n, device=dev), torch.arange(n, device=dev), torch.arange(n, device=dev),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    cv = vals_s[torch.searchsorted(vk_s, morton_encode((cell_coords.unsqueeze(1) + corners).reshape(-1, 3), L))].reshape(-1, 8)
    inside = cv <= 0
    cross = inside[:, elo] != inside[:, ehi]
    hL = 2.0 / n
    active = cross.any(dim=1) | (cv.abs().amin(dim=1) < 0.87 * hL)
    active_coords = cell_coords[active]

    for lev in range(L + 1, level + 1):
        h = 2.0 / (1 << lev)
        child = (active_coords.unsqueeze(1) * 2 + corners).reshape(-1, 3)
        ck = morton_encode((child.unsqueeze(1) + corners).reshape(-1, 3), lev)
        uniq, inv = torch.unique(ck, return_inverse=True)
        pos, hit = _lookup_sorted(vk_s, vals_s, uniq)
        missing = ~hit
        if missing.any():
            mk = uniq[missing]
            mc, _ = morton_decode(mk)
            mv = _eval_chunked(sdf_fn, _world(mc, h))
            allk = torch.cat([vk_s, mk])
            allv = torch.cat([vals_s, mv])
            o = torch.argsort(allk)
            vk_s, vals_s = allk[o], allv[o]
        uvals = vals_s[torch.searchsorted(vk_s, uniq)]
        cv = uvals[inv].reshape(-1, 8)
        inside = cv <= 0
        cross = inside[:, elo] != inside[:, ehi]
        active = cross.any(dim=1) | (cv.abs().amin(dim=1) < 0.87 * h)
        active_coords = child[active]

    h = 2.0 / (1 << level)
    # extraction keeps only sign-change cells; near-zero cells existed solely
    # to drive refinement and carry no Hermite samples
    corner_keys = morton_encode((active_coords.unsqueeze(1) + corners).reshape(-1, 3), level)
    cv = vals_s[torch.searchsorted(vk_s, corner_keys)].reshape(-1, 8)
    inside = cv <= 0
    active_coords = active_coords[(inside[:, elo] != inside[:, ehi]).any(dim=1)]
    C = active_coords.shape[0]
    cell_keys = morton_encode(active_coords, level)
    corner_keys = morton_encode((active_coords.unsqueeze(1) + corners).reshape(-1, 3), level)
    cv = vals_s[torch.searchsorted(vk_s, corner_keys)].reshape(C, 8)
    va, vb = cv[:, elo], cv[:, ehi]
    inside_a, inside_b = va <= 0, vb <= 0
    cross = inside_a != inside_b  # [C, 12]

    lower_coords = active_coords.unsqueeze(1) + corners[elo]  # [C, 12, 3]
    he_keys = edge_key(morton_encode(lower_coords.reshape(-1, 3), level), eax.repeat(C))
    flat_keys = he_keys.reshape(C, _M)[cross]
    edge_keys, inverse = torch.unique(flat_keys, return_inverse=True)
    E = edge_keys.numel()
    cell_edge_slot = torch.full((C, _M), -1, dtype=torch.int64, device=dev)
    cell_edge_slot[cross] = inverse

    edge_lower, edge_axis = edge_split(edge_keys)
    elc, _ = morton_decode(edge_lower)
    ev_a = vals_s[torch.searchsorted(vk_s, edge_lower)]
    up_keys = morton_encode(elc + torch.nn.functional.one_hot(edge_axis, 3).to(torch.int64), level)
    ev_b = vals_s[torch.searchsorted(vk_s, up_keys)]
    axis_vec = torch.nn.functional.one_hot(edge_axis, 3).to(torch.float32)
    # endpoint lerp is biased where the field kinks or saturates (box SDF
    # min-face distance); bisect the true SDF to 2^-16 of the edge
    base = _world(elc, h)
    a_neg = ev_a <= 0
    lo_t = torch.zeros(E, device=dev)
    hi_t = torch.ones(E, device=dev)
    for _ in range(16):
        mid = (lo_t + hi_t) * 0.5
        vm = _eval_chunked(sdf_fn, base + mid.unsqueeze(-1) * h * axis_vec)
        same = (vm <= 0) == a_neg
        lo_t = torch.where(same, mid, lo_t)
        hi_t = torch.where(same, hi_t, mid)
    t = (lo_t + hi_t) * 0.5
    edge_point = base + t.unsqueeze(-1) * h * axis_vec
    if normal_fn is not None:
        edge_normal = _eval_chunked(normal_fn, edge_point)
        edge_normal = edge_normal / edge_normal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    else:
        # fallback for SDFs without analytic normals; eps small enough that the
        # stencil stays on one face except within h/16 of a crease
        edge_normal = _sdf_gradient(sdf_fn, edge_point, h / 16)
    edge_part = (
        _eval_chunked(part_fn, edge_point).to(torch.int64)
        if part_fn is not None
        else torch.zeros(E, dtype=torch.int64, device=dev)
    )
    edge_va_neg = ev_a <= 0

    slot = cell_edge_slot.clamp_min(0)
    samples_p = (edge_point[slot] - _world(active_coords, h).unsqueeze(1)) / h
    samples_n = edge_normal[slot]
    sample_mask = cell_edge_slot >= 0
    part_id = edge_part[slot]
    part_id = torch.where(sample_mask, part_id, torch.zeros_like(part_id))
    first = sample_mask.to(torch.int64).argmax(dim=1)
    cell_part = part_id[torch.arange(C, device=dev), first]

    sheet_id, sheet_count = _split_sheets(samples_p, samples_n, sample_mask)

    return HermiteData(
        level=level, cell_keys=cell_keys, cell_coords=active_coords,
        samples_p=samples_p, samples_n=samples_n, sample_mask=sample_mask,
        sheet_id=sheet_id, sheet_count=sheet_count,
        open_sheet=torch.zeros(C, dtype=torch.bool, device=dev),
        part_id=cell_part,
        edge_lower=elc, edge_axis=edge_axis, edge_point=edge_point,
        edge_normal=edge_normal, edge_va_neg=edge_va_neg, edge_part=edge_part,
        cell_edge_slot=cell_edge_slot,
    )


def sample_heightfield_hermite(
    height_fn: Callable,
    grad_fn: Callable,
    level: int,
    xy_min: tuple[float, float] = (-0.8, -0.8),
    xy_max: tuple[float, float] = (0.8, 0.8),
    device: str = "cuda",
) -> HermiteData:
    """Open-sheet sampler for z = f(x, y): vertical primal edge crossings."""
    dev = torch.device(device)
    h = 2.0 / (1 << level)
    x0 = int(torch.floor(torch.tensor((xy_min[0] + 1) / h)).item())
    x1 = int(torch.ceil(torch.tensor((xy_max[0] + 1) / h)).item())
    y0 = int(torch.floor(torch.tensor((xy_min[1] + 1) / h)).item())
    y1 = int(torch.ceil(torch.tensor((xy_max[1] + 1) / h)).item())
    gx = torch.arange(x0, x1 + 1, device=dev, dtype=torch.int64)
    gy = torch.arange(y0, y1 + 1, device=dev, dtype=torch.int64)
    vxy2 = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), dim=-1).reshape(-1, 2)
    vxy = torch.cat([vxy2, torch.zeros(vxy2.shape[0], 1, device=dev, dtype=torch.int64)], dim=-1)
    wxy = _world(vxy, h)
    hgt = _eval_chunked(height_fn, wxy[:, :2])
    grad = _eval_chunked(grad_fn, wxy[:, :2])
    nrm = torch.cat([-grad, torch.ones_like(grad[:, :1])], dim=-1)
    nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    vkey2 = morton_encode(vxy, 0)  # z=0 plane keys
    order = torch.argsort(vkey2)
    vk2_s, hgt_s, nrm_s = vkey2[order], hgt[order], nrm[order]

    cx = torch.arange(x0, x1, device=dev, dtype=torch.int64)
    cy = torch.arange(y0, y1, device=dev, dtype=torch.int64)
    cols = torch.stack(torch.meshgrid(cx, cy, indexing="ij"), dim=-1).reshape(-1, 2)
    corner_off = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], device=dev, dtype=torch.int64)
    cxy = cols.unsqueeze(1) + corner_off  # [K, 4, 2]
    ck2 = morton_encode(torch.cat([cxy.reshape(-1, 2), torch.zeros(cxy.numel() // 2, 1, device=dev, dtype=torch.int64)], dim=-1), 0)
    pos = torch.searchsorted(vk2_s, ck2)
    ch = hgt_s[pos].reshape(-1, 4)
    cn = nrm_s[pos].reshape(-1, 4, 3)
    zmin, zmax = ch.amin(dim=1), ch.amax(dim=1)

    k_lo = ((zmin + 1) / h).floor().to(torch.int64)
    k_hi = ((zmax + 1) / h).ceil().to(torch.int64)
    counts = (k_hi - k_lo).clamp_min(0)
    total = int(counts.sum().item())
    col_idx = torch.arange(cols.shape[0], device=dev).repeat_interleave(counts)
    starts = torch.cat([torch.zeros(1, device=dev, dtype=torch.int64), counts.cumsum(0)[:-1]])
    k_vals = k_lo[col_idx] + (torch.arange(total, device=dev) - starts[col_idx])
    cell_coords = torch.cat([cols[col_idx], k_vals.unsqueeze(-1)], dim=-1)
    C = cell_coords.shape[0]
    zk_world = -1.0 + k_vals.to(torch.float32) * h

    ch_sel = ch[col_idx]
    cn_sel = cn[col_idx]
    cross = (ch_sel >= zk_world.unsqueeze(-1)) & (ch_sel <= (zk_world + h).unsqueeze(-1))  # [C, 4]
    t = (ch_sel - zk_world.unsqueeze(-1)) / h
    corner_local = corner_off.to(torch.float32).unsqueeze(0).expand(C, 4, 2)
    p_local = torch.cat([corner_local, t.unsqueeze(-1)], dim=-1)

    lower = torch.cat([cxy[col_idx], k_vals.view(C, 1, 1).expand(C, 4, 1)], dim=-1)  # [C, 4, 3]
    he_keys = edge_key(morton_encode(lower.reshape(-1, 3), level), torch.full((lower.shape[0] * 4,), 2, device=dev, dtype=torch.int64))
    flat = he_keys[cross.reshape(-1)]
    edge_keys, inverse = torch.unique(flat, return_inverse=True)
    E = edge_keys.numel()
    cell_edge_slot = torch.full((C, _M), -1, dtype=torch.int64, device=dev)
    cell_edge_slot[:, :4][cross] = inverse

    elc, _ = morton_decode(edge_split(edge_keys)[0])
    # exact crossing point from the first cell that owns each edge
    inv_pos = torch.full((E,), -1, dtype=torch.int64, device=dev)
    inv_pos.scatter_(0, inverse, torch.arange(inverse.numel(), device=dev))
    flat_cell = torch.arange(C, device=dev).unsqueeze(1).expand(C, 4).reshape(-1)[cross.reshape(-1)]
    flat_slot = torch.arange(4, device=dev).unsqueeze(0).expand(C, 4).reshape(-1)[cross.reshape(-1)]
    sel_cell = flat_cell[inv_pos]
    sel_slot = flat_slot[inv_pos]
    edge_point = _world(cell_coords[sel_cell], h) + p_local[sel_cell, sel_slot] * h
    edge_normal = cn_sel[sel_cell, sel_slot]
    edge_axis = torch.full((E,), 2, dtype=torch.int64, device=dev)
    edge_va_neg = torch.ones(E, dtype=torch.bool, device=dev)
    edge_part = torch.zeros(E, dtype=torch.int64, device=dev)

    samples_p = torch.zeros(C, _M, 3, device=dev)
    samples_n = torch.zeros(C, _M, 3, device=dev)
    samples_p[:, :4] = p_local
    samples_n[:, :4] = cn_sel
    sample_mask = cell_edge_slot >= 0
    sheet_id = torch.zeros(C, _M, dtype=torch.int64, device=dev)
    sheet_count = torch.ones(C, dtype=torch.int64, device=dev)

    return HermiteData(
        level=level, cell_keys=morton_encode(cell_coords, level), cell_coords=cell_coords,
        samples_p=samples_p, samples_n=samples_n, sample_mask=sample_mask,
        sheet_id=sheet_id, sheet_count=sheet_count,
        open_sheet=torch.ones(C, dtype=torch.bool, device=dev),
        part_id=torch.zeros(C, dtype=torch.int64, device=dev),
        edge_lower=elc, edge_axis=edge_axis, edge_point=edge_point,
        edge_normal=edge_normal, edge_va_neg=edge_va_neg, edge_part=edge_part,
        cell_edge_slot=cell_edge_slot,
    )


def _split_sheets(
    p: torch.Tensor, n: torch.Tensor, mask: torch.Tensor,
    split_dist: float = 0.4, normal_dot_max: float = -0.2, chunk: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dual-sheet split: farthest sample pair must be spatially separated AND
    carry opposing normals. 90-deg crease normals (dot ~ 0) are one sheet;
    only hollow-shell / near-touching gaps (dot ~ -1) split."""
    C = p.shape[0]
    sheet_id = torch.zeros(C, _M, dtype=torch.int64, device=p.device)
    sheet_count = torch.ones(C, dtype=torch.int64, device=p.device)
    for i in range(0, C, chunk):
        pc, nc, mc = p[i : i + chunk], n[i : i + chunk], mask[i : i + chunk]
        B = pc.shape[0]
        diff = pc.unsqueeze(2) - pc.unsqueeze(1)
        dist = diff.norm(dim=-1)
        valid_pair = mc.unsqueeze(2) & mc.unsqueeze(1)
        ndot_pair = torch.einsum("bmi,bni->bmn", nc, nc)
        # a cell carries two sheets iff ANY separated sample pair opposes;
        # the farthest pair alone can lie on one sheet and miss the split
        opp = (ndot_pair < normal_dot_max) & (dist > split_dist) & valid_pair
        opp = opp & ~torch.eye(_M, dtype=torch.bool, device=p.device).unsqueeze(0)
        split = opp.any(dim=(1, 2))
        flat = opp.reshape(B, -1).to(torch.int64)
        first = flat.argmax(dim=1)
        bi, bj = first // _M, first % _M
        ar = torch.arange(B, device=p.device)
        di = (pc - pc[ar, bi].unsqueeze(1)).norm(dim=-1)
        dj = (pc - pc[ar, bj].unsqueeze(1)).norm(dim=-1)
        sid = torch.where(split.unsqueeze(1) & (dj < di) & mc, torch.ones_like(sheet_id[i : i + chunk]), sheet_id[i : i + chunk])
        sheet_id[i : i + chunk] = sid
        sheet_count[i : i + chunk] = 1 + (split & ((sid == 1).any(dim=1))).to(torch.int64)
    return sheet_id, sheet_count


@dataclass
class SheetInfo:
    verts: torch.Tensor        # [C, 2, 3] cell-local QEF vertices
    normals: torch.Tensor      # [C, 2, 3] mean sheet normals
    centroids: torch.Tensor    # [C, 2, 3]
    ranks: torch.Tensor        # [C, 2] CAD-Snap rank
    valid: torch.Tensor        # [C, 2] bool


def compute_sheets(hd: HermiteData, reg: float = 1e-6) -> SheetInfo:
    p, n, mask = hd.samples_p, hd.samples_n, hd.sample_mask
    m0 = mask & (hd.sheet_id == 0)
    m1 = mask & (hd.sheet_id == 1)
    v0, _ = solve_qef(p, n, m0, reg=reg)
    v1, _ = solve_qef(p, n, m1, reg=reg)
    verts = torch.stack([v0, v1], dim=1)

    w0, w1 = m0.to(p.dtype), m1.to(p.dtype)
    c0 = w0.sum(1).clamp_min(1.0)
    c1 = w1.sum(1).clamp_min(1.0)
    n0 = (n * w0.unsqueeze(-1)).sum(1) / c0.unsqueeze(-1)
    n1 = (n * w1.unsqueeze(-1)).sum(1) / c1.unsqueeze(-1)
    n0 = n0 / n0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n1 = n1 / n1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    normals = torch.stack([n0, n1], dim=1)
    p0 = (p * w0.unsqueeze(-1)).sum(1) / c0.unsqueeze(-1)
    p1 = (p * w1.unsqueeze(-1)).sum(1) / c1.unsqueeze(-1)
    centroids = torch.stack([p0, p1], dim=1)

    info0 = cad_snap_analyze(n, m0)
    info1 = cad_snap_analyze(n, m1)
    ranks = torch.stack([info0.rank, info1.rank], dim=1)
    # rank-3 corners: pin to exact 3-plane intersection
    for s, mm in ((0, m0), (1, m1)):
        corner = ranks[:, s] == 3
        if corner.any():
            verts[corner, s] = snap_corner_vertex(p[corner], n[corner], mm[corner])

    valid = torch.stack([m0.any(dim=1), m1.any(dim=1)], dim=1)
    return SheetInfo(verts=verts, normals=normals, centroids=centroids, ranks=ranks, valid=valid)


@dataclass
class Mesh:
    vertices: torch.Tensor   # [V, 3] world
    faces: torch.Tensor      # [F, 3] int64
    face_part: torch.Tensor  # [F]
    face_open: torch.Tensor  # [F] bool


def dual_contour_mesh(hd: HermiteData, sheets: SheetInfo) -> Mesh:
    """Connect dual vertices around every crossing edge (vectorized)."""
    dev = hd.cell_keys.device
    C, E = hd.cell_keys.numel(), hd.edge_lower.shape[0]
    cell_index = MortonIndex(hd.cell_keys)

    ax = hd.edge_axis
    b_ax = (ax + 1) % 3
    c_ax = (ax + 2) % 3
    eb = torch.nn.functional.one_hot(b_ax, 3).to(torch.int64)
    ec = torch.nn.functional.one_hot(c_ax, 3).to(torch.int64)
    v = hd.edge_lower
    cc = torch.stack([v - eb - ec, v - ec, v, v - eb], dim=1)  # CCW around +axis
    cells = cell_index.query(morton_encode(cc.reshape(-1, 3), hd.level)).reshape(E, 4)

    verts_flat = sheets.verts.reshape(C * 2, 3)
    normals_flat = sheets.normals.reshape(C * 2, 3)
    valid_flat = sheets.valid.reshape(C * 2)
    edge_vid = C * 2 + torch.arange(E, device=dev)

    tris1, tris2, fpart, fopen = [], [], [], []
    for s in range(2):
        # first valid cell as reference when cell 0 absent
        first_valid = (cells >= 0).to(torch.int64).argmax(dim=1)
        ref_cell = cells[torch.arange(E, device=dev), first_valid]
        ref_ok = sheets.valid[ref_cell, s]
        ref_n = sheets.normals[ref_cell, s]

        vids = torch.zeros(E, 4, dtype=torch.int64, device=dev)
        vvalid = torch.zeros(E, 4, dtype=torch.bool, device=dev)
        for j in range(4):
            cj = cells[:, j]
            cjc = cj.clamp_min(0)
            dots = torch.einsum("ekc,ec->ek", sheets.normals[cjc], ref_n)
            dots = torch.where(sheets.valid[cjc], dots, torch.full_like(dots, -2.0))
            best = dots.argmax(dim=1)
            ok = (cj >= 0) & sheets.valid[cjc, best] & ref_ok
            vids[:, j] = torch.where(ok, cjc * 2 + best, torch.zeros_like(vids[:, j]))
            vvalid[:, j] = ok

        # winding: CCW when edge goes inside->outside along +axis, else reversed
        order = torch.where(
            hd.edge_va_neg.unsqueeze(1),
            torch.tensor([0, 1, 2, 3], device=dev).expand(E, 4),
            torch.tensor([3, 2, 1, 0], device=dev).expand(E, 4),
        )
        vids_o = vids.gather(1, order)
        valid_o = vvalid.gather(1, order)
        nvalid = valid_o.sum(dim=1)
        compact = torch.argsort((~valid_o).to(torch.int64), dim=1, stable=True)
        cv = vids_o.gather(1, compact)

        t1 = torch.full((E, 3), -1, dtype=torch.int64, device=dev)
        t2 = torch.full((E, 3), -1, dtype=torch.int64, device=dev)
        q = nvalid == 4
        t1[q] = cv[q][:, :3]
        t2[q] = cv[q][:, [0, 2, 3]]
        tri3 = nvalid == 3
        t1[tri3] = cv[tri3][:, :3]
        b2 = nvalid == 2
        t1[b2, 0] = cv[b2, 0]
        t1[b2, 1] = cv[b2, 1]
        t1[b2, 2] = edge_vid[b2]
        tris1.append(t1)
        tris2.append(t2)
        fpart.append(hd.edge_part)
        fpart.append(hd.edge_part)
        fopen.append(hd.open_sheet[ref_cell])
        fopen.append(hd.open_sheet[ref_cell])

    faces = torch.cat(tris1 + tris2, dim=0)
    face_part = torch.cat(fpart)
    face_open = torch.cat(fopen)
    keep = (faces >= 0).all(dim=1)
    faces, face_part, face_open = faces[keep], face_part[keep], face_open[keep]

    sheet_world = (
        -1.0 + (hd.cell_coords.unsqueeze(1).to(torch.float32) + sheets.verts) * hd.h
    ).reshape(-1, 3)
    vertices = torch.cat([sheet_world, hd.edge_point], dim=0)
    return Mesh(vertices=vertices, faces=faces, face_part=face_part, face_open=face_open)


def leaf_tokens(hd: HermiteData, sheets: SheetInfo, pbr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build wavelet tokens: key = cell_key*2 + sheet, feats[12], covs[3,3].

    pbr: [C, 2, 7] = albedo(3), rough, metal, opacity, emissive.
    """
    C = hd.cell_keys.numel()
    valid = sheets.valid.reshape(-1)
    sheet_ids = torch.arange(2, device=hd.cell_keys.device).expand(C, 2)
    keys = token_key(hd.cell_keys.unsqueeze(1).expand(C, 2), sheet_ids).reshape(-1)[valid]
    oct_n = oct_encode(sheets.normals.reshape(C * 2, 3))
    feats = torch.cat([sheets.verts.reshape(C * 2, 3), oct_n, pbr.reshape(C * 2, 7)], dim=-1)[valid]
    p, n, mask = hd.samples_p, hd.samples_n, hd.sample_mask
    covs = []
    for s in range(2):
        ms = (mask & (hd.sheet_id == s)).to(p.dtype)
        covs.append(torch.einsum("bmi,bmj->bij", n * ms.unsqueeze(-1), n * ms.unsqueeze(-1)))
    covs = torch.stack(covs, dim=1).reshape(C * 2, 3, 3)[valid]
    return keys, feats, covs