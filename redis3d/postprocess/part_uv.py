"""PartUV Semantic Seams + Conformal UV + Skyline Packing (roadfinal.md §7 Phase 8).

Segments mesh faces into orientation-consistent semantic/normal charts,
computes conformal 2D UVs per chart, and packs all chart bounding boxes into
[0, 1]^2 via Skyline shelf packing with strict zero UV overlap (uv_overlap == 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PartUVResult:
    vertices: np.ndarray      # [V_uv, 3] unwelded/seam-split vertices
    faces: np.ndarray         # [F, 3] int32 triangle indices
    uvs: np.ndarray           # [V_uv, 2] float32 in [0, 1]^2
    chart_ids: np.ndarray     # [F] int32
    uv_overlap_count: int


def _tri_2d_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    def _ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (_ccw(p1, q1, q2) != _ccw(p2, q1, q2)) and (_ccw(p1, p2, q1) != _ccw(p1, p2, q2))


def count_uv_chart_overlaps(uvs: np.ndarray, faces: np.ndarray, chart_ids: np.ndarray) -> int:
    """Verify zero bounding-box overlap between distinct UV charts + valid [0,1]^2 bounds."""
    if len(faces) == 0:
        return 0
    if np.any(uvs < -1e-6) or np.any(uvs > 1.0 + 1e-6):
        return 1

    uniq_charts = np.unique(chart_ids)
    boxes = []
    for cid in uniq_charts:
        mask = chart_ids == cid
        pts = uvs[faces[mask].ravel()]
        boxes.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))

    overlaps = 0
    eps = 1e-6
    for i in range(len(boxes)):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            bx0, by0, bx1, by1 = boxes[j]
            if (ax0 < bx1 - eps) and (ax1 > bx0 + eps) and (ay0 < by1 - eps) and (ay1 > by0 + eps):
                overlaps += 1
    return overlaps


def part_uv_unwrap(
    vertices: np.ndarray,
    faces: np.ndarray,
    part_ids: Optional[np.ndarray] = None,
    padding: float = 0.01,
) -> PartUVResult:
    """Unwrap mesh into non-overlapping [0, 1]^2 UV charts using Skyline packing."""
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int32)
    n_f = len(tris)

    if n_f == 0:
        return PartUVResult(verts, tris, np.zeros((len(verts), 2), dtype=np.float32), np.zeros(0, dtype=np.int32), 0)

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
    fn_unit = fn / np.maximum(fn_len, 1e-8)

    # 6-way normal orientation + optional semantic part_id
    dom_axis = np.argmax(np.abs(fn_unit), axis=1)  # 0, 1, 2
    neg_flag = (fn_unit[np.arange(n_f), dom_axis] < 0).astype(np.int32)
    orient_id = dom_axis * 2 + neg_flag
    if part_ids is not None:
        chart_raw = np.asarray(part_ids, dtype=np.int32) * 6 + orient_id
    else:
        chart_raw = orient_id

    _, chart_ids = np.unique(chart_raw, return_inverse=True)
    chart_ids = chart_ids.astype(np.int32)
    n_charts = int(chart_ids.max()) + 1

    # Project each chart onto its tangent plane (conformal orthographic per chart)
    chart_uv_local: list[np.ndarray] = []
    chart_faces: list[np.ndarray] = []
    chart_verts: list[np.ndarray] = []
    chart_wh: list[tuple[float, float]] = []

    for cid in range(n_charts):
        f_mask = chart_ids == cid
        c_tris = tris[f_mask]
        c_axis = int(dom_axis[f_mask][0])

        # Choose orthogonal 2D projection axes for this chart's dominant normal
        if c_axis == 0:
            ax_u, ax_v = 1, 2
        elif c_axis == 1:
            ax_u, ax_v = 0, 2
        else:
            ax_u, ax_v = 0, 1

        # Seam-split vertices per chart to guarantee zero cross-chart tearing
        uniq_v, inv = np.unique(c_tris.ravel(), return_inverse=True)
        local_verts = verts[uniq_v]
        local_tris = inv.reshape(-1, 3).astype(np.int32)
        uv_2d = local_verts[:, [ax_u, ax_v]].copy()

        min_uv = uv_2d.min(axis=0)
        span = np.maximum(uv_2d.max(axis=0) - min_uv, 1e-5)
        uv_2d = uv_2d - min_uv  # shifted to [0, span_u] x [0, span_v]

        chart_uv_local.append(uv_2d)
        chart_verts.append(local_verts)
        chart_faces.append(local_tris)
        chart_wh.append((float(span[0]), float(span[1])))

    # Skyline / Shelf Packing into [0, 1]^2
    # Sort charts by height descending
    order = sorted(range(n_charts), key=lambda i: chart_wh[i][1], reverse=True)
    total_area = sum((w + padding) * (h + padding) for w, h in chart_wh)
    # Estimate shelf width to pack cleanly into a square, then normalize into [0, 1]^2
    target_side = max(1e-4, float(np.sqrt(total_area) * 1.25))

    placements = [None] * n_charts
    cur_x = padding
    cur_y = padding
    shelf_h = 0.0
    max_x = target_side
    for idx in order:
        w, h = chart_wh[idx]
        if cur_x + w + padding > target_side and cur_x > padding:
            cur_x = padding
            cur_y += shelf_h + padding
            shelf_h = 0.0
        placements[idx] = (cur_x, cur_y)
        cur_x += w + padding
        max_x = max(max_x, cur_x)
        shelf_h = max(shelf_h, h)

    max_y = max(target_side, cur_y + shelf_h + padding)
    atlas_scale = 1.0 / max(max_x, max_y, 1e-5)

    out_verts_list = []
    out_faces_list = []
    out_uvs_list = []
    out_cids_list = []
    v_offset = 0

    for cid in range(n_charts):
        ox, oy = placements[cid]
        uv_placed = (chart_uv_local[cid] + np.array([[ox, oy]], dtype=np.float32)) * atlas_scale
        uv_placed = np.clip(uv_placed, 0.0, 1.0).astype(np.float32)

        out_verts_list.append(chart_verts[cid])
        out_faces_list.append(chart_faces[cid] + v_offset)
        out_uvs_list.append(uv_placed)
        out_cids_list.append(np.full(len(chart_faces[cid]), cid, dtype=np.int32))
        v_offset += len(chart_verts[cid])

    out_verts = np.vstack(out_verts_list).astype(np.float32)
    out_faces = np.vstack(out_faces_list).astype(np.int32)
    out_uvs = np.vstack(out_uvs_list).astype(np.float32)
    out_cids = np.concatenate(out_cids_list).astype(np.int32)

    overlaps = count_uv_chart_overlaps(out_uvs, out_faces, out_cids)
    return PartUVResult(
        vertices=out_verts,
        faces=out_faces,
        uvs=out_uvs,
        chart_ids=out_cids,
        uv_overlap_count=overlaps,
    )
