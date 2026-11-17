"""Validated GLB (glTF 2.0 KHR PBR + Skins), OBJ, and USDZ Exporters (roadfinal.md §7 Phase 8)."""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

from redis3d.postprocess.auto_rig import AutoRigResult


def _pad4(data: bytes, pad_byte: bytes = b"\x00") -> bytes:
    rem = len(data) % 4
    if rem == 0:
        return data
    return data + pad_byte * (4 - rem)


def export_glb(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    rig: Optional[AutoRigResult] = None,
    base_color: tuple[float, float, float, float] = (0.75, 0.75, 0.78, 1.0),
    metallic: float = 0.2,
    roughness: float = 0.45,
) -> Path:
    """Export valid glTF 2.0 Binary (.glb) with KHR PBR material and optional Skins."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.uint32)

    if normals is None:
        v_norm = np.zeros_like(verts)
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        np.add.at(v_norm, tris[:, 0], fn)
        np.add.at(v_norm, tris[:, 1], fn)
        np.add.at(v_norm, tris[:, 2], fn)
        n_len = np.linalg.norm(v_norm, axis=1, keepdims=True)
        normals = (v_norm / np.maximum(n_len, 1e-8)).astype(np.float32)
    else:
        normals = np.asarray(normals, dtype=np.float32)

    bin_chunks: list[bytes] = []
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    byte_offset = 0

    def _add_buffer_view(arr: np.ndarray, target: Optional[int] = None) -> int:
        nonlocal byte_offset
        raw = _pad4(arr.tobytes())
        bv_idx = len(buffer_views)
        bv = {"buffer": 0, "byteOffset": byte_offset, "byteLength": len(arr.tobytes())}
        if target is not None:
            bv["target"] = target
        buffer_views.append(bv)
        bin_chunks.append(raw)
        byte_offset += len(raw)
        return bv_idx

    # 0: Indices
    idx_flat = tris.ravel()
    bv_idx = _add_buffer_view(idx_flat, target=34963)
    acc_idx = len(accessors)
    accessors.append(
        {
            "bufferView": bv_idx,
            "componentType": 5125,  # UNSIGNED_INT
            "count": int(idx_flat.size),
            "type": "SCALAR",
        }
    )

    # 1: Positions
    bv_pos = _add_buffer_view(verts, target=34962)
    acc_pos = len(accessors)
    accessors.append(
        {
            "bufferView": bv_pos,
            "componentType": 5126,  # FLOAT
            "count": int(len(verts)),
            "type": "VEC3",
            "min": verts.min(axis=0).tolist(),
            "max": verts.max(axis=0).tolist(),
        }
    )

    # 2: Normals
    bv_nrm = _add_buffer_view(normals, target=34962)
    acc_nrm = len(accessors)
    accessors.append(
        {
            "bufferView": bv_nrm,
            "componentType": 5126,
            "count": int(len(normals)),
            "type": "VEC3",
        }
    )

    attributes = {"POSITION": acc_pos, "NORMAL": acc_nrm}

    if uvs is not None:
        uvs_f = np.asarray(uvs, dtype=np.float32)
        bv_uv = _add_buffer_view(uvs_f, target=34962)
        acc_uv = len(accessors)
        accessors.append(
            {
                "bufferView": bv_uv,
                "componentType": 5126,
                "count": int(len(uvs_f)),
                "type": "VEC2",
            }
        )
        attributes["TEXCOORD_0"] = acc_uv

    nodes = [{"mesh": 0, "name": "Redis3DAsset"}]
    skins = []

    if rig is not None:
        j_arr = np.asarray(rig.skin_joints, dtype=np.uint16)
        w_arr = np.asarray(rig.skin_weights, dtype=np.float32)
        bv_j = _add_buffer_view(j_arr, target=34962)
        acc_j = len(accessors)
        accessors.append(
            {
                "bufferView": bv_j,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": int(len(j_arr)),
                "type": "VEC4",
            }
        )
        bv_w = _add_buffer_view(w_arr, target=34962)
        acc_w = len(accessors)
        accessors.append(
            {
                "bufferView": bv_w,
                "componentType": 5126,  # FLOAT
                "count": int(len(w_arr)),
                "type": "VEC4",
            }
        )
        attributes["JOINTS_0"] = acc_j
        attributes["WEIGHTS_0"] = acc_w

        # Add joint nodes
        joint_node_indices = []
        base_node_idx = len(nodes)
        for ji, jname in enumerate(rig.joint_names):
            pos = rig.joint_positions[ji].tolist()
            node_dict = {"name": jname, "translation": pos}
            children = [
                base_node_idx + ci
                for ci, p_idx in enumerate(rig.joint_parents)
                if p_idx == ji
            ]
            if children:
                node_dict["children"] = children
            nodes.append(node_dict)
            joint_node_indices.append(base_node_idx + ji)

        nodes[0]["skin"] = 0
        skins.append({"joints": joint_node_indices, "skeleton": joint_node_indices[0]})

    bin_blob = b"".join(bin_chunks)
    gltf = {
        "asset": {"version": "2.0", "generator": "Redis3D Pro"},
        "scene": 0,
        "scenes": [{"nodes": [0] + ([1] if rig is not None else [])}],
        "nodes": nodes,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": acc_idx,
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [
            {
                "name": "Redis3DPBR",
                "pbrMetallicRoughness": {
                    "baseColorFactor": list(base_color),
                    "metallicFactor": float(metallic),
                    "roughnessFactor": float(roughness),
                },
            }
        ],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    if skins:
        gltf["skins"] = skins

    json_bytes = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)

    with open(out_path, "wb") as f:
        # GLB Header: magic, version=2, total_length
        f.write(struct.pack("<III", 0x46546C67, 2, total_len))
        # JSON Chunk
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        # BIN Chunk
        f.write(struct.pack("<II", len(bin_blob), 0x004E4942))
        f.write(bin_blob)

    return out_path


def export_obj(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: Optional[np.ndarray] = None,
) -> Path:
    """Export Wavefront OBJ mesh."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Redis3D Pro OBJ Export\n"]
    for v in vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    if uvs is not None:
        for uv in uvs:
            lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for t in faces:
            a, b, c = int(t[0]) + 1, int(t[1]) + 1, int(t[2]) + 1
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c}\n")
    else:
        for t in faces:
            a, b, c = int(t[0]) + 1, int(t[1]) + 1, int(t[2]) + 1
            lines.append(f"f {a} {b} {c}\n")
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def export_usdz(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    """Export minimal valid USDZ (uncompressed ZIP containing ASCII USDA)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pts_str = ", ".join(f"({v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f})" for v in vertices)
    cnt_str = ", ".join(["3"] * len(faces))
    idx_str = ", ".join(str(int(i)) for i in faces.ravel())
    usda = f"""#usda 1.0
(
    defaultPrim = "Redis3DAsset"
    upAxis = "Y"
)
def Mesh "Redis3DAsset"
{{
    point3f[] points = [{pts_str}]
    int[] faceVertexCounts = [{cnt_str}]
    int[] faceVertexIndices = [{idx_str}]
}}
"""
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("asset.usda", usda.encode("utf-8"))
    return out_path
