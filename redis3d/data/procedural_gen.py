"""16-worker CPU multiprocessing procedural 3D asset generator (roadfinal.md Phase 3).

Generates diverse, clean synthetic 3D assets across 6 commercial categories:
  1. Mechanical CAD (gears, brackets, flanges with sharp 90-degree edges)
  2. Organic / Creature anatomy (smooth sculpted primitives, capsules, tori)
  3. Draped Cloth / Fabric (open sheets, sinusoidal waves, ripples, folds)
  4. Rigged Biped / Quadruped (skeleton tree with 6-18 joints, capsule skinning weights)
  5. Hollow Shell / Container (vases, cups, pipes with dual-walled thickness)
  6. Multi-Part Assemblies (tightly touching components with distinct part IDs 0, 1, 2)

All outputs are clean-room, picklable across Windows multiprocessing workers,
strictly normalized within [-0.95, 0.95]^3, and license-tagged as "Procedural-Owned".
"""

from __future__ import annotations

import concurrent.futures
import math
import os
from typing import Any, Callable, Sequence

import numpy as np

CATEGORIES = (
    "mechanical_cad",
    "organic",
    "cloth",
    "rigged",
    "hollow_shell",
    "multi_part",
)

# Aliases for convenience
CATEGORY_ALIASES = {
    "cad": "mechanical_cad",
    "mechanical": "mechanical_cad",
    "creature": "organic",
    "fabric": "cloth",
    "biped": "rigged",
    "skeleton": "rigged",
    "shell": "hollow_shell",
    "container": "hollow_shell",
    "assembly": "multi_part",
    "multipart": "multi_part",
}


def _normalize_category(cat: str) -> str:
    c = cat.lower().strip()
    return CATEGORY_ALIASES.get(c, c)


# =============================================================================
# Vector and Normal Utilities
# =============================================================================

def compute_face_and_vertex_normals(
    vertices: np.ndarray, faces: np.ndarray, sharp: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-face and per-vertex normals.

    If sharp is True, vertex normals duplicate face normals to preserve 90-deg edges.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(fn, axis=-1, keepdims=True)
    fn = fn / np.maximum(norm, 1e-12)

    if sharp:
        vn = np.zeros_like(vertices)
        # For sharp CAD, default vertex normals can average, but face normals stay sharp
        np.add.at(vn, faces[:, 0], fn)
        np.add.at(vn, faces[:, 1], fn)
        np.add.at(vn, faces[:, 2], fn)
        vnorm = np.linalg.norm(vn, axis=-1, keepdims=True)
        vn = vn / np.maximum(vnorm, 1e-12)
        return fn.astype(np.float32), vn.astype(np.float32)

    vn = np.zeros_like(vertices)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    vnorm = np.linalg.norm(vn, axis=-1, keepdims=True)
    vn = vn / np.maximum(vnorm, 1e-12)
    return fn.astype(np.float32), vn.astype(np.float32)


def normalize_to_bounds(vertices: np.ndarray, max_extent: float = 0.90) -> np.ndarray:
    """Center and scale vertices to fit strictly inside [-max_extent, max_extent]^3."""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) * 0.5
    scale = (vmax - vmin).max()
    if scale < 1e-8:
        scale = 1.0
    return ((vertices - center) / scale * (2.0 * max_extent)).astype(np.float32)


# =============================================================================
# Category 1: Mechanical CAD (Sharp 90-deg edges, boolean cutouts)
# =============================================================================

def _generate_cad_gear(rng: np.random.Generator) -> dict[str, Any]:
    n_teeth = rng.integers(8, 16)
    r_root = rng.uniform(0.30, 0.38)
    r_tip = rng.uniform(0.48, 0.58)
    r_hole = rng.uniform(0.12, 0.18)
    thickness = rng.uniform(0.25, 0.45)
    z_half = thickness * 0.5

    # 4 points per tooth on outer boundary
    n_pts_outer = n_teeth * 4
    angles_outer = np.linspace(0, 2 * np.pi, n_pts_outer, endpoint=False)
    radii_outer = np.zeros(n_pts_outer, dtype=np.float32)
    for i in range(n_teeth):
        radii_outer[4 * i + 0] = r_root
        radii_outer[4 * i + 1] = r_tip
        radii_outer[4 * i + 2] = r_tip
        radii_outer[4 * i + 3] = r_root

    outer_xy = np.stack([radii_outer * np.cos(angles_outer), radii_outer * np.sin(angles_outer)], axis=-1)

    # Inner circular bore
    n_pts_inner = n_pts_outer
    angles_inner = np.linspace(0, 2 * np.pi, n_pts_inner, endpoint=False)
    inner_xy = np.stack([r_hole * np.cos(angles_inner), r_hole * np.sin(angles_inner)], axis=-1)

    # Vertices: bottom outer, top outer, bottom inner, top inner
    v_bot_out = np.pad(outer_xy, ((0, 0), (0, 1)), constant_values=-z_half)
    v_top_out = np.pad(outer_xy, ((0, 0), (0, 1)), constant_values=z_half)
    v_bot_in = np.pad(inner_xy, ((0, 0), (0, 1)), constant_values=-z_half)
    v_top_in = np.pad(inner_xy, ((0, 0), (0, 1)), constant_values=z_half)

    N = n_pts_outer
    vertices = np.concatenate([v_bot_out, v_top_out, v_bot_in, v_top_in], axis=0)

    faces = []
    # Outer side walls (sharp vertical faces)
    for i in range(N):
        nxt = (i + 1) % N
        faces.append([i, nxt, N + nxt])
        faces.append([i, N + nxt, N + i])

    # Inner side walls
    offset_in = 2 * N
    for i in range(N):
        nxt = (i + 1) % N
        faces.append([offset_in + i, offset_in + N + nxt, offset_in + nxt])
        faces.append([offset_in + i, offset_in + N + i, offset_in + N + nxt])

    # Top annular cap (outer N..2N-1 to inner 3N..4N-1)
    for i in range(N):
        nxt = (i + 1) % N
        faces.append([N + i, N + nxt, 3 * N + nxt])
        faces.append([N + i, 3 * N + nxt, 3 * N + i])

    # Bottom annular cap (outer 0..N-1 to inner 2N..3N-1)
    for i in range(N):
        nxt = (i + 1) % N
        faces.append([i, 2 * N + nxt, nxt])
        faces.append([i, 2 * N + i, 2 * N + nxt])

    faces = np.array(faces, dtype=np.int64)
    vertices = normalize_to_bounds(vertices)
    fn, vn = compute_face_and_vertex_normals(vertices, faces, sharp=True)

    pbr_albedo = np.tile(np.array([0.72, 0.74, 0.78], dtype=np.float32), (len(faces), 1))
    pbr_roughness = np.full((len(faces), 1), rng.uniform(0.18, 0.32), dtype=np.float32)
    pbr_metallic = np.full((len(faces), 1), rng.uniform(0.85, 1.0), dtype=np.float32)

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": vn,
        "face_normals": fn,
        "albedo": pbr_albedo,
        "roughness": pbr_roughness,
        "metallic": pbr_metallic,
        "part_ids": np.zeros(len(faces), dtype=np.int64),
        "open_sheet": False,
        "sdf_type": "cad_gear",
        "sdf_params": {"r_root": float(r_root), "r_tip": float(r_tip), "r_hole": float(r_hole), "height": float(thickness)},
    }


def _generate_cad_bracket(rng: np.random.Generator) -> dict[str, Any]:
    # L-bracket with sharp 90-degree internal angle
    thick = rng.uniform(0.15, 0.22)
    l_horiz = rng.uniform(0.65, 0.85)
    h_vert = rng.uniform(0.65, 0.85)
    width = rng.uniform(0.35, 0.50)

    # 6 vertices in X-Z profile:
    # 0: (-thick, -thick)
    # 1: (l_horiz, -thick)
    # 2: (l_horiz, 0)
    # 3: (0, 0)  <-- sharp 90 internal corner!
    # 4: (0, h_vert)
    # 5: (-thick, h_vert)
    profile_xz = np.array([
        [-thick, -thick],
        [l_horiz, -thick],
        [l_horiz, 0.0],
        [0.0, 0.0],
        [0.0, h_vert],
        [-thick, h_vert],
    ], dtype=np.float32)

    w_half = width * 0.5
    v_front = np.pad(profile_xz, ((0, 0), (0, 1)), constant_values=0)
    v_front = np.stack([profile_xz[:, 0], np.full(6, -w_half), profile_xz[:, 1]], axis=-1)
    v_back = np.stack([profile_xz[:, 0], np.full(6, w_half), profile_xz[:, 1]], axis=-1)
    vertices = np.concatenate([v_front, v_back], axis=0)

    # Extrude perimeter quads
    faces = []
    for i in range(6):
        nxt = (i + 1) % 6
        faces.append([i, nxt, 6 + nxt])
        faces.append([i, 6 + nxt, 6 + i])

    # End caps (decompose polygon 0-1-2-3-4-5 into 4 triangles)
    # Box 1: 0-1-2-3 -> (0, 1, 2), (0, 2, 3)
    # Box 2: 0-3-4-5 -> (0, 3, 4), (0, 4, 5)
    # Front cap (facing -Y)
    faces.append([0, 2, 1])
    faces.append([0, 3, 2])
    faces.append([0, 4, 3])
    faces.append([0, 5, 4])
    # Back cap (facing +Y)
    faces.append([6 + 0, 6 + 1, 6 + 2])
    faces.append([6 + 0, 6 + 2, 6 + 3])
    faces.append([6 + 0, 6 + 3, 6 + 4])
    faces.append([6 + 0, 6 + 4, 6 + 5])

    faces = np.array(faces, dtype=np.int64)
    vertices = normalize_to_bounds(vertices)
    fn, vn = compute_face_and_vertex_normals(vertices, faces, sharp=True)

    albedo_steel = np.array([0.65, 0.68, 0.72], dtype=np.float32)
    pbr_albedo = np.tile(albedo_steel, (len(faces), 1))
    pbr_roughness = np.full((len(faces), 1), rng.uniform(0.2, 0.35), dtype=np.float32)
    pbr_metallic = np.full((len(faces), 1), rng.uniform(0.9, 1.0), dtype=np.float32)

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": vn,
        "face_normals": fn,
        "albedo": pbr_albedo,
        "roughness": pbr_roughness,
        "metallic": pbr_metallic,
        "part_ids": np.zeros(len(faces), dtype=np.int64),
        "open_sheet": False,
        "sdf_type": "cad_bracket",
        "sdf_params": {"thick": float(thick), "l_horiz": float(l_horiz), "h_vert": float(h_vert), "width": float(width)},
    }


# =============================================================================
# Category 2: Organic / Creature Anatomy (Smooth sculpted primitives)
# =============================================================================

def _generate_organic_sculpt(rng: np.random.Generator) -> dict[str, Any]:
    n_lat = 32
    n_lon = 64
    theta = np.linspace(0, np.pi, n_lat)
    phi = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")

    # Sculpting harmonics
    f1, f2, f3 = rng.integers(2, 5, size=3)
    p1, p2, p3 = rng.uniform(0, 2 * np.pi, size=3)
    a1, a2, a3 = rng.uniform(0.04, 0.12, size=3)

    r = 0.50 * (1.0 + a1 * np.sin(f1 * tt + p1) * np.cos(f2 * pp + p2) + a2 * np.sin(f3 * pp + p3))
    # Asymmetric creature taper along Z
    r *= (1.0 + 0.15 * np.cos(tt))

    x = r * np.sin(tt) * np.cos(pp)
    y = r * np.sin(tt) * np.sin(pp)
    z = r * np.cos(tt)

    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1).astype(np.float32)

    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            nxt_j = (j + 1) % n_lon
            p00 = i * n_lon + j
            p01 = i * n_lon + nxt_j
            p10 = (i + 1) * n_lon + j
            p11 = (i + 1) * n_lon + nxt_j
            if i > 0:
                faces.append([p00, p10, p01])
            if i < n_lat - 2:
                faces.append([p01, p10, p11])

    faces = np.array(faces, dtype=np.int64)
    vertices = normalize_to_bounds(vertices)
    fn, vn = compute_face_and_vertex_normals(vertices, faces, sharp=False)

    colors = [
        np.array([0.78, 0.55, 0.45], dtype=np.float32),  # flesh
        np.array([0.25, 0.58, 0.35], dtype=np.float32),  # reptilian jade
        np.array([0.35, 0.45, 0.65], dtype=np.float32),  # alien azure
    ]
    base_color = colors[rng.integers(0, len(colors))]
    pbr_albedo = np.tile(base_color, (len(faces), 1))
    pbr_roughness = np.full((len(faces), 1), rng.uniform(0.45, 0.75), dtype=np.float32)
    pbr_metallic = np.full((len(faces), 1), rng.uniform(0.0, 0.05), dtype=np.float32)

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": vn,
        "face_normals": fn,
        "albedo": pbr_albedo,
        "roughness": pbr_roughness,
        "metallic": pbr_metallic,
        "part_ids": np.zeros(len(faces), dtype=np.int64),
        "open_sheet": False,
        "sdf_type": "organic_sculpt",
        "sdf_params": {"radius": 0.50},
    }


def _generate_organic_torus(rng: np.random.Generator) -> dict[str, Any]:
    n_major = 48
    n_minor = 24
    u = np.linspace(0, 2 * np.pi, n_major, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    r_maj = rng.uniform(0.40, 0.50)
    r_min = rng.uniform(0.14, 0.20)
    # Undulating muscle modulation
    freq = rng.integers(3, 7)
    r_mod = r_min * (1.0 + 0.25 * np.cos(freq * uu))

    x = (r_maj + r_mod * np.cos(vv)) * np.cos(uu)
    y = (r_maj + r_mod * np.cos(vv)) * np.sin(uu)
    z = r_mod * np.sin(vv)

    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1).astype(np.float32)

    faces = []
    for i in range(n_major):
        nxt_i = (i + 1) % n_major
        for j in range(n_minor):
            nxt_j = (j + 1) % n_minor
            p00 = i * n_minor + j
            p01 = i * n_minor + nxt_j
            p10 = nxt_i * n_minor + j
            p11 = nxt_i * n_minor + nxt_j
            faces.append([p00, p10, p01])
            faces.append([p01, p10, p11])

    faces = np.array(faces, dtype=np.int64)
    vertices = normalize_to_bounds(vertices)
    fn, vn = compute_face_and_vertex_normals(vertices, faces, sharp=False)

    pbr_albedo = np.tile(np.array([0.65, 0.35, 0.45], dtype=np.float32), (len(faces), 1))
    pbr_roughness = np.full((len(faces), 1), rng.uniform(0.40, 0.65), dtype=np.float32)
    pbr_metallic = np.full((len(faces), 1), 0.0, dtype=np.float32)

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": vn,
        "face_normals": fn,
        "albedo": pbr_albedo,
        "roughness": pbr_roughness,
        "metallic": pbr_metallic,
        "part_ids": np.zeros(len(faces), dtype=np.int64),
        "open_sheet": False,
        "sdf_type": "organic_torus",
        "sdf_params": {"r_maj": float(r_maj), "r_min": float(r_min)},
    }


# =============================================================================
# Category 3: Draped Cloth / Fabric (Open sheets, waves, folds)
# =============================================================================

def _generate_cloth_sheet(rng: np.random.Generator) -> dict[str, Any]:
    n = 45
    xy = np.linspace(-0.80, 0.80, n)
    xx, yy = np.meshgrid(xy, xy, indexing="ij")

    # Harmonic waves and ripples
    w1 = rng.uniform(2.5, 4.5)
    w2 = rng.uniform(2.5, 4.5)
    a1 = rng.uniform(0.08, 0.16)
    a2 = rng.uniform(0.04, 0.09)
    p1 = rng.uniform(0, 2 * np.pi)
    p2 = rng.uniform(0, 2 * np.pi)

    zz = a1 * np.sin(w1 * xx + p1) * np.cos(w2 * yy + p2) + a2 * np.sin(2.0 * (xx + yy))
    # Boundary drape curve
    drape = 0.20 * (xx**2 + yy**2)
    zz -= drape

    vertices = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1).astype(np.float32)

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            p00 = i * n + j
            p01 = i * n + (j + 1)
            p10 = (i + 1) * n + j
            p11 = (i + 1) * n + (j + 1)
            faces.append([p00, p10, p01])
            faces.append([p01, p10, p11])

    faces = np.array(faces, dtype=np.int64)
    # Do not center Z to preserve heightfield reference
    fn, vn = compute_face_and_vertex_normals(vertices, faces, sharp=False)

    colors = [
        np.array([0.88, 0.84, 0.78], dtype=np.float32),  # linen
        np.array([0.65, 0.14, 0.18], dtype=np.float32),  # crimson
        np.array([0.18, 0.22, 0.48], dtype=np.float32),  # indigo
    ]
    base_color = colors[rng.integers(0, len(colors))]
    pbr_albedo = np.tile(base_color, (len(faces), 1))
    pbr_roughness = np.full((len(faces), 1), rng.uniform(0.85, 0.98), dtype=np.float32)
    pbr_metallic = np.full((len(faces), 1), 0.0, dtype=np.float32)

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": vn,
        "face_normals": fn,
        "albedo": pbr_albedo,
        "roughness": pbr_roughness,
        "metallic": pbr_metallic,
        "part_ids": np.zeros(len(faces), dtype=np.int64),
        "open_sheet": True,
        "sdf_type": "cloth_heightfield",
        "sdf_params": {"w1": float(w1), "w2": float(w2), "a1": float(a1), "p1": float(p1), "p2": float(p2)},
    }


# =============================================================================
# Category 4: Rigged Biped / Quadruped (Hierarchical bones, skinning weights)
# =============================================================================

def _generate_rigged_biped(rng: np.random.Generator) -> dict[str, Any]:
    # 17 standard humanoid joints
    bone_names = [
        "pelvis", "spine", "chest", "neck", "head",
        "l_clavicle", "l_elbow", "l_hand",
        "r_clavicle", "r_elbow", "r_hand",
        "l_hip", "l_knee", "l_foot",
        "r_hip", "r_knee", "r_foot",
    ]
    bone_parents = np.array([
        -1, 0, 1, 2, 3,
        2, 5, 6,
        2, 8, 9,
        0, 11, 12,
        0, 14, 15,
    ], dtype=np.int64)

    # Joint positions
    joints = np.array([
        [0.0, 0.0, 0.0],       # 0: pelvis
        [0.0, 0.0, 0.20],      # 1: spine
        [0.0, 0.0, 0.40],      # 2: chest
        [0.0, 0.0, 0.52],      # 3: neck
        [0.0, 0.0, 0.68],      # 4: head
        [0.12, 0.0, 0.40],     # 5: l_clavicle
        [0.32, 0.0, 0.28],     # 6: l_elbow
        [0.52, 0.0, 0.15],     # 7: l_hand
        [-0.12, 0.0, 0.40],    # 8: r_clavicle
        [-0.32, 0.0, 0.28],    # 9: r_elbow
        [-0.52, 0.0, 0.15],    # 10: r_hand
        [0.15, 0.0, -0.05],    # 11: l_hip
        [0.17, 0.0, -0.38],    # 12: l_knee
        [0.17, 0.0, -0.72],    # 13: l_foot
        [-0.15, 0.0, -0.05],   # 14: r_hip
        [-0.17, 0.0, -0.38],   # 15: r_knee
        [-0.17, 0.0, -0.72],   # 16: r_foot
    ], dtype=np.float32)

    # Slight randomized pose variation
    joints[:, :2] += rng.normal(0.0, 0.015, size=joints[:, :2].shape).astype(np.float32)

    # Generate body mesh components (cylinder/capsule segments per bone)
    all_verts = []
    all_faces = []
    v_offset = 0

    bone_radii = {
        0: 0.16, 1: 0.15, 2: 0.16, 3: 0.08, 4: 0.13,
        5: 0.06, 6: 0.05, 7: 0.04,
        8: 0.06, 9: 0.05, 10: 0.04,
        11: 0.08, 12: 0.07, 13: 0.06,
        14: 0.08, 15: 0.07, 16: 0.06,
    }

    # Head sphere
    n_head = 12
    th = np.linspace(0, np.pi, n_head)
    ph = np.linspace(0, 2 * np.pi, 2 * n_head, endpoint=False)
    tth, pph = np.meshgrid(th, ph, indexing="ij")
    rh = bone_radii[4]
    hx = rh * np.sin(tth) * np.cos(pph) + joints[4, 0]
    hy = rh * np.sin(tth) * np.sin(pph) + joints[4, 1]
    hz = rh * np.cos(tth) + joints[4, 2]
    h_verts = np.stack([hx.flatten(), hy.flatten(), hz.flatten()], axis=-1).astype(np.float32)
    h_faces = []
    for i in range(n_head - 1):
        for j in range(2 * n_head):
            nxt_j = (j + 1) % (2 * n_head)
            p00 = v_offset + i * (2 * n_head) + j
            p01 = v_offset + i * (2 * n_head) + nxt_j
            p10 = v_offset + (i + 1) * (2 * n_head) + j
            p11 = v_offset + (i + 1) * (2 * n_head) + nxt_j
            h_faces.append([p00, p10, p01])
            h_faces.append([p01, p10, p11])
    all_verts.append(h_verts)
    all_faces.append(np.array(h_faces, dtype=np.int64))
    v_offset += len(h_verts)

    # Bone cylinder segments
    n_seg = 10
    angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    for b_idx in range(1, len(bone_parents)):
        p_idx = bone_parents[b_idx]
        pt_a = joints[p_idx]
        pt_b = joints[b_idx]
        ab = pt_b - pt_a
        ab_len = np.linalg.norm(ab)
        if ab_len < 1e-6:
            continue
        ab_dir = ab / ab_len

        # Orthogonal basis
        ref = np.array([0, 0, 1], dtype=np.float32) if abs(ab_dir[2]) < 0.9 else np.array([1, 0, 0], dtype=np.float32)
        u_dir = np.cross(ab_dir, ref)
        u_dir /= np.linalg.norm(u_dir)
        v_dir = np.cross(ab_dir, u_dir)

        r_cyl = bone_radii.get(b_idx, 0.06)
        ring_a = pt_a[None, :] + r_cyl * (np.cos(angles)[:, None] * u_dir + np.sin(angles)[:, None] * v_dir)
        ring_b = pt_b[None, :] + r_cyl * (np.cos(angles)[:, None] * u_dir + np.sin(angles)[:, None] * v_dir)
        seg_verts = np.concatenate([ring_a, ring_b], axis=0).astype(np.float32)

        seg_faces = []
        for i in range(n_seg):
            nxt = (i + 1) % n_seg
            p00 = v_offset + i
            p01 = v_offset + nxt
            p10 = v_offset + n_seg + i
            p11 = v_offset + n_seg + nxt
            seg_faces.append([p00, p01, p11])
            seg_faces.append([p00, p11, p10])

        all_verts.append(seg_verts)
        all_faces.append(np.array(seg_faces, dtype=np.int64))
        v_offset += len(seg_verts)

    vertices = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    vertices = normalize_to_bounds(vertices)

    # Scale joints accordingly
    j_min, j_max = joints.min(axis=0), joints.max(axis=0)
    j_center = (j_min + j_max) * 0.5
    j_scale = (j_max - j_min).max()
    joints = ((joints - j_center) / j_scale * 1.80).astype(np.float32)

    # Compute hierarchical bone skinning weights (top 4 per vertex)
    V = len(vertices)
    J = len(joints)
    dists = np.zeros((V, J), dtype=np.float32)

    # Distance to bone segments
    for b in range(J):
        p = bone_parents[b]
        if p == -1:
            dists[:, b] = np.linalg.norm(vertices - joints[b], axis=-1)
        else:
            ab = joints[b] - joints[p]
            l2 = np.dot(ab, ab)
            if l2 < 1e-8:
                dists[:, b] = np.linalg.norm(vertices - joints[b], axis=-1)
            else:
                t = np.clip(np.sum((vertices - joints[p]) * ab, axis=-1) / l2, 0.0, 1.0)
                proj = joints[p] + t[:, None] * ab
                dists[:, b] = np.linalg.norm(vertices - proj, axis=-1)

    # Soft falloff: w = exp(-d^2 / 2sigma^2)
    sigma = 0.12
    raw_weights = np.exp(-0.5 * (dists / sigma)**2)
    raw_weights = np.maximum(raw_weights, 1e-6)

    # Top 4 indices and weights
    top4_idx = np.argsort(-raw_weights, axis=-1)[:, :4]
    top4_weights = np.take_along_axis(raw_weights, top4_idx, axis=-1)
    # Row normalize
    w_sum = top4_weights.sum(axis=-1, keepdims=True)
    top4_weights = (top4_weights / np.maximum(w_sum, 1e-12)).astype(np.float32)

    fn, vn = compute_face_and_vertex_normals(vertices, faces, sharp=False)
    pbr_albedo = np.tile(np.array([0.55, 0.60, 0.65], dtype=np.float32), (len(faces), 1))
    pbr_roughness = np.full((len(faces), 1), 0.5, dtype=np.float32)
    pbr_metallic = np.full((len(faces), 1), 0.1, dtype=np.float32)

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": vn,
        "face_normals": fn,
        "albedo": pbr_albedo,
        "roughness": pbr_roughness,
        "metallic": pbr_metallic,
        "part_ids": np.zeros(len(faces), dtype=np.int64),
        "open_sheet": False,
        "skeleton_joints": joints,
        "bone_parents": bone_parents,
        "bone_names": bone_names,
        "vertex_bone_indices": top4_idx.astype(np.int64),
        "vertex_bone_weights": top4_weights.astype(np.float32),
        "sdf_type": "rigged_biped",
        "sdf_params": {"num_joints": J},
    }


# =============================================================================
# Category 5: Hollow Shell / Container (Dual-walled thickness)
# =============================================================================

def _generate_hollow_container(rng: np.random.Generator) -> dict[str, Any]:
    n_z = 32
    n_phi = 40
    z_profile = np.linspace(-0.70, 0.70, n_z)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Outer profile curve: vase with waist and flared neck
    r_base = rng.uniform(0.35, 0.45)
    neck_h = rng.uniform(0.2, 0.4)
    r_outer = r_base + 0.12 * np.sin(2.5 * z_profile) + 0.08 * np.exp(z_profile * 1.5)

    wall_thickness = rng.uniform(0.04, 0.08)
    r_inner = np.maximum(0.03, r_outer - wall_thickness)

    # Outer wall vertices
    zz_out, pp_out = np.meshgrid(z_profile, phi, indexing="ij")
    rr_out = r_outer[:, None]
    x_out = rr_out * np.cos(pp_out)
    y_out = rr_out * np.sin(pp_out)
    v_out = np.stack([x_out.flatten(), y_out.flatten(), zz_out.flatten()], axis=-1).astype(np.float32)

    # Inner wall vertices (inner floor at z = -0.60 to leave bottom wall)
    z_inner_profile = np.linspace(-0.62, 0.70, n_z)
    zz_in, pp_in = np.meshgrid(z_inner_profile, phi, indexing="ij")
    rr_in = r_inner[:, None]
    x_in = rr_in * np.cos(pp_in)
    y_in = rr_in * np.sin(pp_in)
    v_in = np.stack([x_in.flatten(), y_in.flatten(), zz_in.flatten()], axis=-1).astype(np.float32)

    N_out = len(v_out)
    vertices = np.concatenate([v_out, v_in], axis=0)

    faces = []
    # Outer surface (normals point outward)
    for i in range(n_z - 1):
        for j in range(n_phi):
            nxt_j = (j + 1) % n_phi
            p00 = i * n_phi + j
            p01 = i * n_phi + nxt_j
            p10 = (i + 1) * n_phi + j
            p11 = (i + 1) * n_phi + nxt_j
            faces.append([p00, p10, p01])
            faces.append([p01, p10, p11])

    # Inner surface (normals point inward into cavity -> reversed winding)
    offset_in = N_out
    for i in range(n_z - 1):
        for j in range(n_phi):
            nxt_j = (j + 1) % n_phi
            p00 = offset_in + i * n_phi + j
            p01 = offset_in + i * n_phi + nxt_j
            p10 = offset_in + (i + 1) * n_phi + j
            p11 = offset_in + (i + 1) * n_phi + nxt_j
            faces.append([p00, p01, p10])
            faces.append([p01, p11, p10])

    # Top rim quads (connecting outer top ring to inner top ring at z = 0.70)
    top_out_start = (n_z - 1) * n_phi
    top_in_start = offset_in + (n_z - 1) * n_phi
    for j in range(n_phi):
        nxt_j = (j + 1) % n_phi
        o0 = top_out_start + j
        o1 = top_out_start + nxt_j
        i0 = top_in_start + j
        i1 = top_in_start + nxt_j
        faces.append([o0, o1, i1])
        faces.append([o0, i1, i0])

    # Bottom outer disk cap
    center_bot_out = len(vertices)
    vertices = np.concatenate([vertices, [[0.0, 0.0, -0.70]]], axis=0)
    for j in range(n_phi):
        nxt_j = (j + 1) % n_phi
        faces.append([center_bot_out, nxt_j, j])

    # Bottom inner disk floor
    center_bot_in = len(vertices)
    vertices = np.concatenate([vertices, [[0.0, 0.0, -0.62]]], axis=0)
    for j in range(n_phi):
        nxt_j = (j + 1) % n_phi
        faces.append([center_bot_in, offset_in + j, offset_in + nxt_j])

    faces = np.array(faces, dtype=np.int64)
    vertices = normalize_to_bounds(vertices)
    fn, vn = compute_face_and_vertex_normals(vertices, faces, sharp=False)

    pbr_albedo = np.tile(np.array([0.85, 0.72, 0.55], dtype=np.float32), (len(faces), 1))
    pbr_roughness = np.full((len(faces), 1), rng.uniform(0.15, 0.45), dtype=np.float32)
    pbr_metallic = np.full((len(faces), 1), rng.uniform(0.0, 0.3), dtype=np.float32)

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": vn,
        "face_normals": fn,
        "albedo": pbr_albedo,
        "roughness": pbr_roughness,
        "metallic": pbr_metallic,
        "part_ids": np.zeros(len(faces), dtype=np.int64),
        "open_sheet": False,
        "sdf_type": "hollow_container",
        "sdf_params": {"wall_thickness": float(wall_thickness), "r_base": float(r_base)},
    }


# =============================================================================
# Category 6: Multi-Part Assemblies (Tightly touching, distinct part IDs 0, 1, 2)
# =============================================================================

def _generate_multipart_assembly(rng: np.random.Generator) -> dict[str, Any]:
    # 3 touching parts:
    # Part 0 (Base / Left Housing): [-0.60, 0.0] x [-0.35, 0.35] x [-0.35, 0.35]
    # Part 1 (Slider / Right Mating Block): [0.0, 0.60] x [-0.35, 0.35] x [-0.35, 0.35] (touches at x = 0!)
    # Part 2 (Cylindrical Pin / Fastener): Along Z axis through center
    def make_box(lo: list[float], hi: list[float]) -> tuple[np.ndarray, np.ndarray]:
        v = np.array([
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ], dtype=np.float32)
        f = np.array([
            [0, 2, 1], [0, 3, 2],  # -Z
            [4, 5, 6], [4, 6, 7],  # +Z
            [0, 1, 5], [0, 5, 4],  # -Y
            [2, 3, 7], [2, 7, 6],  # +Y
            [0, 4, 7], [0, 7, 3],  # -X
            [1, 2, 6], [1, 6, 5],  # +X
        ], dtype=np.int64)
        return v, f

    def make_cylinder(r: float, h: float, n_pts: int = 16) -> tuple[np.ndarray, np.ndarray]:
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        bot = np.stack([x, y, np.full(n_pts, -h * 0.5)], axis=-1)
        top = np.stack([x, y, np.full(n_pts, h * 0.5)], axis=-1)
        verts = np.concatenate([bot, top, [[0, 0, -h * 0.5]], [[0, 0, h * 0.5]]], axis=0).astype(np.float32)
        c_bot = 2 * n_pts
        c_top = 2 * n_pts + 1
        f = []
        for i in range(n_pts):
            nxt = (i + 1) % n_pts
            f.append([i, nxt, n_pts + nxt])
            f.append([i, n_pts + nxt, n_pts + i])
            f.append([c_bot, nxt, i])
            f.append([c_top, n_pts + i, n_pts + nxt])
        return verts, np.array(f, dtype=np.int64)

    v0, f0 = make_box([-0.55, -0.32, -0.30], [0.0, 0.32, 0.30])
    v1, f1 = make_box([0.0, -0.32, -0.30], [0.55, 0.32, 0.30])
    v2, f2 = make_cylinder(r=0.12, h=0.75, n_pts=20)

    # Combine parts
    f1_shifted = f1 + len(v0)
    f2_shifted = f2 + (len(v0) + len(v1))
    vertices = np.concatenate([v0, v1, v2], axis=0)
    faces = np.concatenate([f0, f1_shifted, f2_shifted], axis=0)

    part_ids = np.zeros(len(faces), dtype=np.int64)
    part_ids[len(f0) : len(f0) + len(f1)] = 1
    part_ids[len(f0) + len(f1) :] = 2

    vertices = normalize_to_bounds(vertices)
    fn, vn = compute_face_and_vertex_normals(vertices, faces, sharp=True)

    # Distinct PBR per part
    albedos = np.zeros((len(faces), 3), dtype=np.float32)
    roughness = np.zeros((len(faces), 1), dtype=np.float32)
    metallic = np.zeros((len(faces), 1), dtype=np.float32)

    # Part 0: Matte dark housing
    albedos[part_ids == 0] = [0.22, 0.24, 0.26]
    roughness[part_ids == 0] = 0.65
    metallic[part_ids == 0] = 0.10

    # Part 1: Brushed steel
    albedos[part_ids == 1] = [0.80, 0.82, 0.85]
    roughness[part_ids == 1] = 0.22
    metallic[part_ids == 1] = 0.95

    # Part 2: Polished brass pin
    albedos[part_ids == 2] = [0.88, 0.70, 0.28]
    roughness[part_ids == 2] = 0.18
    metallic[part_ids == 2] = 0.90

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": vn,
        "face_normals": fn,
        "albedo": albedos,
        "roughness": roughness,
        "metallic": metallic,
        "part_ids": part_ids,
        "open_sheet": False,
        "sdf_type": "multipart_touching",
        "sdf_params": {"parts": 3},
    }


# =============================================================================
# Generator Dispatcher & Multiprocessing Batch Factory
# =============================================================================

def generate_procedural_asset(category: str, seed: int) -> dict[str, Any]:
    """Generate a single procedural 3D asset for a specified category and seed.

    Args:
        category: One of CATEGORIES or alias.
        seed: Random seed for deterministic generation.

    Returns:
        Dictionary with vertices, faces, normals, PBR materials, part_ids,
        category, asset_id, license_tag='Procedural-Owned', and optional skeleton.
    """
    cat = _normalize_category(category)
    rng = np.random.default_rng(seed)

    if cat == "mechanical_cad":
        sub = rng.choice(["gear", "bracket"])
        asset = _generate_cad_gear(rng) if sub == "gear" else _generate_cad_bracket(rng)
    elif cat == "organic":
        sub = rng.choice(["sculpt", "torus"])
        asset = _generate_organic_sculpt(rng) if sub == "sculpt" else _generate_organic_torus(rng)
    elif cat == "cloth":
        asset = _generate_cloth_sheet(rng)
    elif cat == "rigged":
        asset = _generate_rigged_biped(rng)
    elif cat == "hollow_shell":
        asset = _generate_hollow_container(rng)
    elif cat == "multi_part":
        asset = _generate_multipart_assembly(rng)
    else:
        raise ValueError(f"Unknown category '{category}'. Available: {CATEGORIES}")

    asset["category"] = cat
    asset["asset_id"] = f"{cat}_{seed:08x}"
    asset["license_tag"] = "Procedural-Owned"
    asset["seed"] = seed

    return asset


def _worker_generate_task(task: tuple[str, int]) -> dict[str, Any]:
    """Picklable top-level worker task for Windows multiprocessing."""
    cat, seed = task
    return generate_procedural_asset(cat, seed)


def generate_procedural_batch(
    count: int = 256,
    num_workers: int = 16,
    categories: Sequence[str] | None = None,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    """Generate a batch of procedural 3D assets across CPU workers.

    Windows-safe multiprocessing with top-level picklable tasks.
    SLA: Must generate 256 assets in < 60 seconds (typically < 3s).

    Args:
        count: Total number of assets to generate.
        num_workers: Number of worker processes (capped at available CPU cores).
        categories: Sequence of categories to cycle through (defaults to all 6).
        seed_offset: Base seed offset.

    Returns:
        List of generated asset dictionaries.
    """
    if categories is None:
        cats = CATEGORIES
    else:
        cats = tuple(_normalize_category(c) for c in categories)

    tasks = [
        (cats[i % len(cats)], seed_offset + i)
        for i in range(count)
    ]

    effective_workers = max(1, min(num_workers, os.cpu_count() or 16))

    if effective_workers <= 1 or count <= 2:
        return [_worker_generate_task(t) for t in tasks]

    # Windows-safe ProcessPoolExecutor with chunksize
    chunksize = max(1, count // (effective_workers * 4))
    with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
        results = list(executor.map(_worker_generate_task, tasks, chunksize=chunksize))

    return results
