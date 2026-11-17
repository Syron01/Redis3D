"""Deterministic half-edge topology repair (roadfinal.md Phase 2, MeshRepair).

Pipeline: weld -> degenerate/duplicate drop -> half-edge build -> patch
flood-fill (manifold edges only) -> non-manifold vertex fan split ->
consistent outward orientation -> debris removal (< 0.1% of largest patch
area; area works for open patches too) -> minimal hole bridging (centroid
fan). Every step is deterministic: sorted keys, fixed traversal order, no
randomness. Optional bilateral Sym-Lock averages mirror pairs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class RepairReport:
    vertices_in: int = 0
    faces_in: int = 0
    vertices_out: int = 0
    faces_out: int = 0
    welded: int = 0
    degenerate_removed: int = 0
    duplicate_faces_removed: int = 0
    patches_split: int = 0
    pinch_vertices_split: int = 0
    faces_flipped: int = 0
    debris_faces_removed: int = 0
    holes_bridged: int = 0
    hole_faces_added: int = 0
    sym_locked_pairs: int = 0


def _as_numpy(vertices, faces):
    v = vertices.detach().cpu().numpy().astype(np.float64) if hasattr(vertices, "detach") else np.asarray(vertices, dtype=np.float64)
    f = faces.detach().cpu().numpy().astype(np.int64) if hasattr(faces, "detach") else np.asarray(faces, dtype=np.int64)
    return v, f


def weld_vertices(v: np.ndarray, eps: float = 1e-7):
    """Merge vertices on an eps grid; cluster-mean position, stable order."""
    keys = np.floor(v / eps + 0.5).astype(np.int64)
    uniq, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.sort(first)
    remap = np.searchsorted(order, first[inverse]).astype(np.int64)
    nv = np.zeros((len(order), 3), dtype=np.float64)
    np.add.at(nv, remap, v)
    cnt = np.bincount(remap, minlength=len(order)).astype(np.float64)
    nv /= cnt[:, None]
    return nv, remap, len(v) - len(order)


def build_halfedges(f: np.ndarray, n_verts: int) -> dict:
    """Half-edge structure: origin, dest, face, next, twin (-1 = boundary)."""
    F = len(f)
    origin = f.reshape(-1).copy()
    dest = np.roll(f, -1, axis=1).reshape(-1).copy()
    face = np.repeat(np.arange(F, dtype=np.int64), 3)
    nxt = np.arange(3 * F, dtype=np.int64).reshape(F, 3)[:, [1, 2, 0]].reshape(-1)
    lo = np.minimum(origin, dest)
    hi = np.maximum(origin, dest)
    keys = lo.astype(np.int64) * n_verts + hi.astype(np.int64)
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    twin = np.full(3 * F, -1, dtype=np.int64)
    s = np.argsort(inv, kind="stable")
    si = inv[s]
    starts = np.flatnonzero(np.concatenate([[True], si[1:] != si[:-1]]))
    sizes = np.diff(np.concatenate([starts, [len(si)]]))
    two = starts[sizes == 2]
    a, b = s[two], s[two + 1]
    twin[a] = b
    twin[b] = a
    return {"origin": origin, "dest": dest, "face": face, "next": nxt,
            "twin": twin, "edge_count": counts[inv], "edge_id": inv}


def _corner_neighbors(he: dict) -> tuple[np.ndarray, np.ndarray]:
    """Two fan-neighbors per face-corner, winding-independent.

    Corner h is the outgoing half-edge at origin[h] in face(h). Its fan
    neighbors are the corners at the same vertex in the faces across each
    of the two incident edges. In a triangle the corner at vertex a of the
    neighboring face is q itself when origin[q]==a (flipped winding), else
    next[q]. -1 marks a boundary side.
    """
    twin, nxt, origin = he["twin"], he["next"], he["origin"]
    q1 = twin
    q2 = twin[nxt[nxt]]
    n1 = np.where(q1 >= 0, np.where(origin[np.maximum(q1, 0)] == origin, q1, nxt[np.maximum(q1, 0)]), -1)
    n2 = np.where(q2 >= 0, np.where(origin[np.maximum(q2, 0)] == origin, q2, nxt[np.maximum(q2, 0)]), -1)
    return n1, n2


def _corner_labels(he: dict) -> np.ndarray:
    H = len(he["origin"])
    n1, n2 = _corner_neighbors(he)
    labels = np.arange(H)
    for _ in range(64):
        prop = labels.copy()
        for nb in (n1, n2):
            ok = nb >= 0
            m = np.minimum(labels[ok], labels[nb[ok]])
            np.minimum.at(prop, np.nonzero(ok)[0], m)
            np.minimum.at(prop, nb[ok], m)
        if np.array_equal(prop, labels):
            break
        labels = prop
    return labels


def vertex_fan_counts(f: np.ndarray, n_verts: int) -> np.ndarray:
    """Number of edge-connected face fans around each vertex.

    Min-label propagation over the corner adjacency graph; each orbit is
    one fan (cycle if closed, path if boundary). Winding-independent, so
    flipped faces cannot split or merge fans.
    """
    he = build_halfedges(f, n_verts)
    H = 3 * len(f)
    labels = _corner_labels(he)
    origin = he["origin"]
    pair = origin.astype(np.int64) * H + labels
    upair = np.unique(pair)
    fans = np.zeros(n_verts, dtype=np.int64)
    np.add.at(fans, upair // H, 1)
    return fans


def signed_volume(v: np.ndarray, f: np.ndarray) -> float:
    if len(f) == 0:
        return 0.0
    t = v[f]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def mesh_health(v: np.ndarray, f: np.ndarray) -> dict:
    """Independent manifold / watertight / orientation audit."""
    if len(f) == 0:
        return {"watertight": False, "edge_manifold": False, "vertex_manifold": False,
                "oriented": False, "components": 0, "volume": 0.0,
                "boundary_edges": 0, "nonmanifold_edges": 0}
    he = build_halfedges(f, len(v))
    boundary = int((he["edge_count"] == 1).sum())
    nonmanifold = int((he["edge_count"] > 2).sum() // 3)
    fans = vertex_fan_counts(f, len(v))
    used = np.unique(f)
    vertex_ok = bool((fans[used] == 1).all())
    directed = f.reshape(-1).astype(np.int64) * len(v) + np.roll(f, -1, axis=1).reshape(-1)
    _, dcounts = np.unique(directed, return_counts=True)
    orient_ok = bool((dcounts <= 1).all())
    keys, lo, hi = _edge_keys(f, len(v))
    rows = np.concatenate([lo, hi])
    cols = np.concatenate([hi, lo])
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(v), len(v)))
    ncomp, _ = connected_components(g, directed=False)
    return {"watertight": boundary == 0 and nonmanifold == 0,
            "edge_manifold": nonmanifold == 0, "vertex_manifold": vertex_ok,
            "oriented": orient_ok, "components": int(ncomp),
            "volume": float(signed_volume(v, f)),
            "boundary_edges": boundary, "nonmanifold_edges": nonmanifold}


def _edge_keys(f: np.ndarray, n_verts: int):
    lo = np.minimum(f, np.roll(f, -1, axis=1))
    hi = np.maximum(f, np.roll(f, -1, axis=1))
    keys = lo.astype(np.int64) * n_verts + hi.astype(np.int64)
    return keys.reshape(-1), lo.reshape(-1), hi.reshape(-1)


def _face_patches(f: np.ndarray, n_verts: int):
    """Flood-fill faces through manifold (2-face) edges only."""
    he = build_halfedges(f, n_verts)
    twin, face = he["twin"], he["face"]
    linked = twin >= 0
    rows = np.concatenate([face[linked], face[twin[linked]]])
    cols = np.concatenate([face[twin[linked]], face[linked]])
    F = len(f)
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(F, F))
    ncomp, labels = connected_components(g, directed=False)
    return labels, ncomp


def _split_pinch_vertices(v: np.ndarray, f: np.ndarray):
    """Duplicate vertices with >1 edge-connected fan (pinch points)."""
    fans = vertex_fan_counts(f, len(v))
    candidates = np.nonzero(fans > 1)[0]
    if len(candidates) == 0:
        return v, f, 0
    he = build_halfedges(f, len(v))
    labels = _corner_labels(he)
    origin, face = he["origin"], he["face"]
    new_v = list(v)
    f = f.copy()
    splits = 0
    for vert in candidates.tolist():
        hs = np.nonzero(origin == vert)[0]
        groups = {}
        for h in hs.tolist():
            groups.setdefault(int(labels[h]), []).append(h)
        if len(groups) <= 1:
            continue
        keys = sorted(groups)
        for gk in keys[1:]:
            nid = len(new_v)
            new_v.append(v[vert])
            for h in groups[gk]:
                fi = int(face[h])
                f[fi][f[fi] == vert] = nid
            splits += 1
    return np.asarray(new_v, dtype=np.float64), f, splits


def _orient_patch(v: np.ndarray, f: np.ndarray):
    """BFS flip propagation over twins so shared edges run opposite; outward by volume."""
    F = len(f)
    he = build_halfedges(f, len(v))
    twin, face, origin = he["twin"], he["face"], he["origin"]
    same_dir = np.zeros(3 * F, dtype=bool)
    linked = twin >= 0
    same_dir[linked] = origin[twin[linked]] == origin[linked]
    flip = np.zeros(F, dtype=bool)
    visited = np.zeros(F, dtype=bool)
    for start in range(F):
        if visited[start]:
            continue
        visited[start] = True
        dq = deque([start])
        while dq:
            fi = dq.popleft()
            for h in range(3 * fi, 3 * fi + 3):
                t = int(twin[h])
                if t < 0:
                    continue
                fj = int(face[t])
                if visited[fj]:
                    continue
                # original twin same-direction => exactly one side must flip
                flip[fj] = flip[fi] ^ bool(same_dir[h])
                visited[fj] = True
                dq.append(fj)
    nflip = int(flip.sum())
    f[flip] = f[flip][:, [0, 2, 1]]
    if signed_volume(v, f) < 0:
        f = f[:, [0, 2, 1]]
        nflip += F
    return f, nflip


def _boundary_loops(f: np.ndarray, n_verts: int):
    he = build_halfedges(f, n_verts)
    b = he["twin"] < 0
    d_lo = he["origin"][b]
    d_hi = he["dest"][b]
    nxt = {}
    for a, bb in zip(d_lo.tolist(), d_hi.tolist()):
        nxt[a] = bb
    loops = []
    used = set()
    for a in sorted(nxt):
        if a in used:
            continue
        loop = [a]
        used.add(a)
        bb = nxt[a]
        while bb != a and bb not in used:
            loop.append(bb)
            used.add(bb)
            bb = nxt.get(bb, a)
        if bb == a and len(loop) >= 3:
            loops.append(loop)
    return loops


def sym_lock(v: np.ndarray, f: np.ndarray, axis: int = 0, eps: float = 1e-6):
    """Bilateral mirror lock: average each vertex with its mirror partner."""
    out = v.copy()
    pairs = 0
    seen = set()
    for i in range(len(v)):
        if i in seen:
            continue
        mir = v[i].copy()
        mir[axis] = -mir[axis]
        d = np.abs(v - mir).max(axis=1)
        j = int(np.argmin(d))
        if d[j] >= eps:
            continue
        if j == i:
            out[i, axis] = 0.0
            seen.add(i)
            continue
        avg = (v[i] + v[j]) / 2
        mag = (abs(v[i, axis]) + abs(v[j, axis])) / 2
        out[i] = avg
        out[i, axis] = mag if v[i, axis] >= 0 else -mag
        out[j] = avg
        out[j, axis] = -out[i, axis]
        seen.add(i)
        seen.add(j)
        pairs += 1
    return out, f, pairs


def repair_mesh(vertices, faces, weld_eps: float = 1e-7, debris_frac: float = 1e-3,
                close_holes: bool = True, sym_lock_axis: int | None = None):
    """Full deterministic repair. Returns (vertices, faces, RepairReport) numpy."""
    v, f = _as_numpy(vertices, faces)
    rep = RepairReport(vertices_in=len(v), faces_in=len(f))
    if sym_lock_axis is not None:
        v, f, rep.sym_locked_pairs = sym_lock(v, f, axis=sym_lock_axis)

    v, remap, rep.welded = weld_vertices(v, weld_eps)
    f = remap[f]

    good = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    t = v[f]
    area2 = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    good &= area2 > 1e-14
    rep.degenerate_removed = int((~good).sum())
    f = f[good]
    if len(f) == 0:
        rep.vertices_out, rep.faces_out = len(v), 0
        return v, f, rep

    # exact duplicate faces (either winding) create 3-face non-manifold edges
    srt = np.sort(f, axis=1)
    _, keep_idx = np.unique(srt, axis=0, return_index=True)
    keep_idx.sort()
    rep.duplicate_faces_removed = len(f) - len(keep_idx)
    f = f[keep_idx]

    labels, npatch = _face_patches(f, len(v))
    rep.patches_split = npatch
    out_v, out_f, patch_areas = [], [], []
    for p in range(npatch):
        pf = f[labels == p]
        used = np.unique(pf)
        remap_p = np.full(len(v), -1, dtype=np.int64)
        remap_p[used] = np.arange(len(used))
        pv = v[used]
        pf = remap_p[pf]
        pv, pf, ns = _split_pinch_vertices(pv, pf)
        rep.pinch_vertices_split += ns
        pf, nflip = _orient_patch(pv, pf)
        rep.faces_flipped += nflip
        t = pv[pf]
        patch_areas.append(float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() * 0.5))
        out_v.append(pv)
        out_f.append(pf)

    amax = max(patch_areas) if patch_areas else 0.0
    keep_v, keep_f = [], []
    base = 0
    for p in range(npatch):
        if amax > 0 and patch_areas[p] < debris_frac * amax:
            rep.debris_faces_removed += len(out_f[p])
        else:
            keep_v.append(out_v[p])
            keep_f.append(out_f[p] + base)
            base += len(out_v[p])
    if not keep_v:
        rep.vertices_out, rep.faces_out = 0, 0
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), rep
    v = np.concatenate(keep_v)
    f = np.concatenate(keep_f)

    if close_holes:
        keys, lo, hi = _edge_keys(f, len(v))
        rows = np.concatenate([lo, hi])
        cols = np.concatenate([hi, lo])
        g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(v), len(v)))
        ncomp, clabel = connected_components(g, directed=False)
        new_v = list(v)
        add_f = []
        for c in range(ncomp):
            cf = f[clabel[f[:, 0]] == c]
            for loop in _boundary_loops(cf, len(v)):
                cen = v[loop].mean(axis=0)
                cid = len(new_v)
                new_v.append(cen)
                for a, b in zip(loop, loop[1:] + loop[:1]):
                    add_f.append([cid, b, a])
                rep.holes_bridged += 1
        if add_f:
            rep.hole_faces_added = len(add_f)
            v = np.asarray(new_v, dtype=np.float64)
            f = np.concatenate([f, np.asarray(add_f, dtype=np.int64)])
            keys, lo, hi = _edge_keys(f, len(v))
            rows = np.concatenate([lo, hi])
            cols = np.concatenate([hi, lo])
            g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(v), len(v)))
            ncomp, clabel = connected_components(g, directed=False)
            for c in range(ncomp):
                m = clabel[f[:, 0]] == c
                if signed_volume(v, f[m]) < 0:
                    f[m] = f[m][:, [0, 2, 1]]
                    rep.faces_flipped += int(m.sum())

    rep.vertices_out, rep.faces_out = len(v), len(f)
    return v, f, rep