"""Phase 2 acceptance gate — deterministic topology repair (roadfinal.md Phase 2).

"Inject synthetic defects (flipped normals, open seams, non-manifold pinch
vertices, floating dust triangles) into 50 meshes -> repair restores 100%
manifold + watertight (50/50) in < 100 ms/mesh."

10 clean-room analytic shapes x 5 seeded defect recipes = 50 meshes.
Inputs are CUDA tensors; the whole gate runs inside VRAMBudgetGuard(2048).
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import pytest
import torch

from redis3d.core.hardware_guard import VRAMBudgetGuard
from redis3d.math.topology_repair import (
    build_halfedges,
    mesh_health,
    repair_mesh,
    signed_volume,
    sym_lock,
    vertex_fan_counts,
    weld_vertices,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="real GPU required")
DEV = "cuda"


# ---------- clean-room closed-mesh generators ----------

def _outward(v, f):
    if signed_volume(v, f) < 0:
        f = f[:, [0, 2, 1]]
    return v, f


def box_mesh():
    v = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.4, 0.4) for z in (-0.3, 0.3)], dtype=np.float64)
    f = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ], dtype=np.int64)
    return _outward(v, f)


def tetra_mesh():
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64) * 0.4
    f = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int64)
    return _outward(v, f)


def octa_mesh():
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64) * 0.45
    f = np.array([
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ], dtype=np.int64)
    return _outward(v, f)


def icosphere_mesh(subdiv: int):
    t = (1 + 5 ** 0.5) / 2
    v = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    v *= 0.4
    for _ in range(subdiv):
        mid = {}
        nv = list(v)
        nf = []
        for a, b, c in f.tolist():
            ids = []
            for x, y in ((a, b), (b, c), (c, a)):
                key = (min(x, y), max(x, y))
                if key not in mid:
                    p = (v[x] + v[y]) / 2
                    p = p / np.linalg.norm(p) * 0.4
                    mid[key] = len(nv)
                    nv.append(p)
                ids.append(mid[key])
            ab, bc, ca = ids
            nf.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        v = np.asarray(nv, dtype=np.float64)
        f = np.asarray(nf, dtype=np.int64)
    return _outward(v, f)


def uv_sphere_mesh(rings: int, segs: int):
    verts = [np.array([0.0, 0.4, 0.0])]
    for j in range(1, rings):
        th = np.pi * j / rings
        for i in range(segs):
            ph = 2 * np.pi * i / segs
            verts.append(0.4 * np.array([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)]))
    verts.append(np.array([0.0, -0.4, 0.0]))
    v = np.asarray(verts, dtype=np.float64)
    top, bot = 0, len(v) - 1
    f = []
    for i in range(segs):
        a, b = 1 + i, 1 + (i + 1) % segs
        f.append([top, b, a])
    for j in range(rings - 2):
        r0, r1 = 1 + j * segs, 1 + (j + 1) * segs
        for i in range(segs):
            i2 = (i + 1) % segs
            f.append([r0 + i, r0 + i2, r1 + i2])
            f.append([r0 + i, r1 + i2, r1 + i])
    rlast = 1 + (rings - 2) * segs
    for i in range(segs):
        a, b = rlast + i, rlast + (i + 1) % segs
        f.append([bot, a, b])
    return _outward(v, np.asarray(f, dtype=np.int64))


def torus_mesh(major: int, minor: int, R: float = 0.3, r: float = 0.12):
    v = []
    for i in range(major):
        u = 2 * np.pi * i / major
        for j in range(minor):
            w = 2 * np.pi * j / minor
            v.append([(R + r * np.cos(w)) * np.cos(u), r * np.sin(w), (R + r * np.cos(w)) * np.sin(u)])
    v = np.asarray(v, dtype=np.float64)
    f = []
    for i in range(major):
        i2 = (i + 1) % major
        for j in range(minor):
            j2 = (j + 1) % minor
            a, b, c, d = i * minor + j, i2 * minor + j, i2 * minor + j2, i * minor + j2
            f.append([a, b, c])
            f.append([a, c, d])
    return _outward(v, np.asarray(f, dtype=np.int64))


def cylinder_mesh(segs: int):
    v = []
    for y in (0.35, -0.35):
        for i in range(segs):
            ph = 2 * np.pi * i / segs
            v.append([0.25 * np.cos(ph), y, 0.25 * np.sin(ph)])
    v.append([0.0, 0.35, 0.0])
    v.append([0.0, -0.35, 0.0])
    v = np.asarray(v, dtype=np.float64)
    top, bot = 2 * segs, 2 * segs + 1
    f = []
    for i in range(segs):
        i2 = (i + 1) % segs
        f.append([i, i2, segs + i2])
        f.append([i, segs + i2, segs + i])
        f.append([top, i2, i])
        f.append([bot, segs + i, segs + i2])
    return _outward(v, np.asarray(f, dtype=np.int64))


def cone_mesh(segs: int):
    v = [[0.0, 0.45, 0.0]]
    for i in range(segs):
        ph = 2 * np.pi * i / segs
        v.append([0.3 * np.cos(ph), -0.3, 0.3 * np.sin(ph)])
    v.append([0.0, -0.3, 0.0])
    v = np.asarray(v, dtype=np.float64)
    apex, cen = 0, len(v) - 1
    f = []
    for i in range(segs):
        i2 = (i + 1) % segs
        f.append([apex, 1 + i2, 1 + i])
        f.append([cen, 1 + i, 1 + i2])
    return _outward(v, np.asarray(f, dtype=np.int64))


def prism_mesh():
    v = []
    for y in (0.3, -0.3):
        for k in range(3):
            ph = 2 * np.pi * k / 3
            v.append([0.35 * np.cos(ph), y, 0.35 * np.sin(ph)])
    v = np.asarray(v, dtype=np.float64)
    f = [[0, 2, 1], [3, 4, 5],
         [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]]
    return _outward(v, np.asarray(f, dtype=np.int64))


SHAPES = [
    ("box", box_mesh),
    ("tetra", tetra_mesh),
    ("octa", octa_mesh),
    ("prism", prism_mesh),
    ("ico1", lambda: icosphere_mesh(1)),
    ("ico2", lambda: icosphere_mesh(2)),
    ("ico3", lambda: icosphere_mesh(3)),
    ("uv_sphere", lambda: uv_sphere_mesh(8, 12)),
    ("torus", lambda: torus_mesh(12, 8)),
    ("cylinder", lambda: cylinder_mesh(10)),
]
# cone kept as an extra unit-check shape; 10 shapes x 5 recipes = 50 gate meshes


# ---------- seeded defect injection ----------

def _edge_adjacency(f):
    adj = {}
    for fi, (a, b, c) in enumerate(f.tolist()):
        for x, y in ((a, b), (b, c), (c, a)):
            adj.setdefault((min(x, y), max(x, y)), []).append(fi)
    return adj


def inject_defects(v, f, recipe: int, seed: int):
    rng = np.random.default_rng(seed)
    v, f = v.copy(), f.copy()
    do_flip = recipe in (0, 1, 3)
    do_seam = recipe in (1, 3, 4)
    do_pinch = recipe in (2, 3)
    do_dust = recipe in (0, 2, 3, 4)
    do_dup = recipe == 4

    if do_flip:
        k = max(1, len(f) // 10)
        sel = rng.choice(len(f), size=k, replace=False)
        f[sel] = f[sel][:, [0, 2, 1]]

    if do_seam:
        adj = _edge_adjacency(f)
        face_nb = {}
        for fis in adj.values():
            for x in fis:
                for y in fis:
                    if x != y:
                        face_nb.setdefault(x, set()).add(y)
        start = int(rng.integers(len(f)))
        seen = {start}
        dq = deque([start])
        # hole stays small relative to mesh resolution: a flat centroid-fan
        # bridge cannot reproduce curvature across a large hole
        k = min(8, max(1, len(f) // 16))
        while dq and len(seen) < k:
            cur = dq.popleft()
            for nb in sorted(face_nb.get(cur, ())):
                if nb not in seen and len(seen) < k:
                    seen.add(nb)
                    dq.append(nb)
        keep = np.array(sorted(set(range(len(f))) - seen), dtype=np.int64)
        f = f[keep]

    if do_pinch:
        # topological pinch: split one vertex's face fan into two coincident
        # copies (geometry unchanged, vertex becomes non-manifold). Merging
        # distant vertices instead would move geometry and no repair could
        # restore the original volume.
        from redis3d.math.topology_repair import _corner_neighbors, build_halfedges
        he = build_halfedges(f, len(v))
        n1, n2 = _corner_neighbors(he)
        origin, face = he["origin"], he["face"]
        counts = np.bincount(origin, minlength=len(v))
        vert = int(np.argmax(counts))
        hs = np.nonzero(origin == vert)[0]
        if len(hs) >= 4:
            hset = set(hs.tolist())
            order = [int(hs[0])]
            prev, cur = -1, int(hs[0])
            while True:
                nxts = [int(x) for x in (n1[cur], n2[cur])
                        if x >= 0 and int(x) in hset and int(x) != prev]
                if not nxts or nxts[0] == order[0]:
                    break
                nxt = nxts[0]
                order.append(nxt)
                prev, cur = cur, nxt
            k = len(order) // 2
            nid = len(v)
            v = np.concatenate([v, v[vert:vert + 1]])
            for h in order[k:]:
                fi = int(face[h])
                f[fi][f[fi] == vert] = nid

    if do_dust:
        cen = v.mean(axis=0) + np.array([3.0, 3.0, 3.0])
        base = len(v)
        dv = []
        df = []
        for t in range(3):
            off = cen + rng.normal(size=3) * 0.05
            p0 = off
            p1 = off + np.array([1e-3, 0.0, 0.0])
            p2 = off + np.array([0.0, 1e-3, 0.0])
            dv.extend([p0, p1, p2])
            df.append([base + 3 * t, base + 3 * t + 1, base + 3 * t + 2])
        v = np.concatenate([v, np.asarray(dv)])
        f = np.concatenate([f, np.asarray(df, dtype=np.int64)])

    if do_dup:
        di = int(rng.integers(len(f)))
        f = np.concatenate([f, f[di:di + 1]])

    return v, f


# ---------- unit tests ----------

def test_cube_halfedge_structure_and_euler():
    v, f = box_mesh()
    he = build_halfedges(f, len(v))
    assert len(he["origin"]) == 36
    assert (he["twin"] >= 0).all(), "closed cube must have no boundary half-edges"
    assert (he["twin"][he["twin"]] == np.arange(36)).all()
    assert (he["origin"][he["twin"]] == he["dest"]).all()
    assert (he["edge_count"] == 2).all()
    n_edges = len(np.unique(np.minimum(he["origin"], he["dest"]) * len(v) + np.maximum(he["origin"], he["dest"])))
    assert len(v) - n_edges + len(f) == 2, "Euler characteristic must be 2"
    h = mesh_health(v, f)
    assert h["watertight"] and h["edge_manifold"] and h["vertex_manifold"] and h["oriented"]
    assert h["volume"] > 0


def test_generators_are_clean_closed_manifolds():
    for name, gen in SHAPES + [("cone", lambda: cone_mesh(10))]:
        v, f = gen()
        h = mesh_health(v, f)
        assert h["watertight"], f"{name} not watertight: {h}"
        assert h["vertex_manifold"], f"{name} vertex non-manifold"
        assert h["oriented"] and h["volume"] > 0, f"{name} orientation/volume"


def test_inside_out_sphere_repaired():
    v, f = icosphere_mesh(2)
    f = f[:, [0, 2, 1]]
    assert signed_volume(v, f) < 0
    rv, rf, rep = repair_mesh(v, f)
    h = mesh_health(rv, rf)
    assert h["watertight"] and h["volume"] > 0
    assert rep.faces_flipped > 0


def test_open_box_hole_bridged():
    v, f = box_mesh()
    f = f[2:]  # remove one quad (2 triangles) -> square hole
    rv, rf, rep = repair_mesh(v, f)
    h = mesh_health(rv, rf)
    assert h["watertight"] and h["vertex_manifold"] and h["oriented"]
    assert rep.holes_bridged == 1 and rep.hole_faces_added >= 3
    assert abs(h["volume"] - signed_volume(*box_mesh())) < 1e-9


def test_pinch_vertex_two_closed_fans_detected_and_split():
    v, f = icosphere_mesh(1)
    d = np.linalg.norm(v - v[0], axis=1)
    a = int(np.argmax(d))
    d2 = np.linalg.norm(v - v[a], axis=1)
    b = int(np.argmax(d2))
    v[b] = v[a]
    vw, remap, _ = weld_vertices(v)
    fw = remap[f]
    fans = vertex_fan_counts(fw, len(vw))
    merged = remap[a]
    assert fans[merged] == 2, "two closed fans must be detected (E==F counting would miss this)"
    rv, rf, rep = repair_mesh(v, f)
    h = mesh_health(rv, rf)
    assert h["watertight"] and h["vertex_manifold"]
    assert rep.pinch_vertices_split >= 1


def test_sym_lock_averages_mirror_pairs():
    v, f = box_mesh()
    v[:, 0] += np.where(v[:, 0] > 0, 0.01, -0.004)  # break bilateral symmetry
    lv, lf, pairs = sym_lock(v, f, axis=0, eps=0.05)
    assert pairs == 4
    assert np.abs(lv[:, 0]).max() > 0
    for i in range(len(lv)):
        mir = lv[i].copy()
        mir[0] = -mir[0]
        d = np.abs(lv - mir).max(axis=1)
        assert d.min() < 1e-12, "every vertex must have an exact mirror partner"


def test_repair_is_deterministic():
    v, f = icosphere_mesh(2)
    dv, df = inject_defects(v, f, recipe=3, seed=42)
    v1, f1, r1 = repair_mesh(dv, df)
    v2, f2, r2 = repair_mesh(dv, df)
    assert np.array_equal(v1, v2) and np.array_equal(f1, f2)
    assert r1 == r2


# ---------- the 50-mesh constitution gate ----------

def test_gate_50_defective_meshes_restored():
    results = []
    with VRAMBudgetGuard(2048) as guard:
        for si, (name, gen) in enumerate(SHAPES):
            cv, cf = gen()
            clean_vol = signed_volume(cv, cf)
            cv_t = torch.tensor(cv, device=DEV)
            cf_t = torch.tensor(cf, device=DEV)
            for recipe in range(5):
                dv, df = inject_defects(cv, cf, recipe, seed=1000 + si * 5 + recipe)
                dv_t = torch.tensor(dv, device=DEV)
                df_t = torch.tensor(df, device=DEV)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                rv, rf, rep = repair_mesh(dv_t, df_t)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                h = mesh_health(rv, rf)
                ok = (h["watertight"] and h["edge_manifold"] and h["vertex_manifold"]
                      and h["oriented"] and h["volume"] > 0)
                vol_err = abs(h["volume"] - clean_vol) / clean_vol
                results.append((name, recipe, ok, vol_err, dt_ms, rep))
                assert ok, f"{name} recipe{recipe} not manifold/watertight: {h}"
                assert vol_err < 0.05, f"{name} recipe{recipe} volume drift {vol_err:.3%}"
                assert dt_ms < 100.0, f"{name} recipe{recipe} took {dt_ms:.1f} ms >= 100 ms"
    n_ok = sum(1 for r in results if r[2])
    worst_ms = max(r[4] for r in results)
    mean_ms = float(np.mean([r[4] for r in results]))
    worst_vol = max(r[3] for r in results)
    print(f"\nPhase 2 gate: {n_ok}/50 manifold+watertight, "
          f"mean={mean_ms:.1f} ms/mesh, worst={worst_ms:.1f} ms, "
          f"max volume drift={worst_vol:.3%}, "
          f"peak VRAM={guard.peak_mb:.1f} MB, shared spill={guard.shared_spill_mb:.1f} MB")
    assert n_ok == 50