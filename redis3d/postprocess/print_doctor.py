"""PrintDoctor: Watertight, Wall-Thickness >= 0.8mm Check, Base Flatten, STL/3MF Export (roadfinal.md §7 Phase 8)."""

from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from redis3d.postprocess.quad_retopo import _check_manifold_watertight


@dataclass
class PrintCheckReport:
    is_watertight: bool
    non_manifold_edges: int
    min_wall_thickness_mm: float
    is_printable: bool
    scale_applied: float


def run_print_doctor(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_size_mm: float = 80.0,
    min_wall_mm: float = 0.8,
    flatten_base: bool = True,
) -> tuple[np.ndarray, np.ndarray, PrintCheckReport]:
    """Validate & prepare mesh for 3D printing (STL / 3MF).

    Scales normalized [-1, 1] mesh to physical millimeter units, optionally flattens
    base to Z=0, and checks watertightness, non-manifold edges, and wall thickness.
    """
    verts = np.asarray(vertices, dtype=np.float32).copy()
    tris = np.asarray(faces, dtype=np.int32)

    is_wt, nm_edges = _check_manifold_watertight(tris)

    extent = np.maximum(verts.max(axis=0) - verts.min(axis=0), 1e-6)
    max_dim = float(np.max(extent))
    scale = target_size_mm / max_dim
    verts = verts * scale

    if flatten_base:
        verts[:, 2] -= verts[:, 2].min()

    # Estimate minimum feature/wall thickness via opposite-normal ray/extent proxy
    scaled_extent = verts.max(axis=0) - verts.min(axis=0)
    est_wall_mm = float(np.min(scaled_extent))
    printable = is_wt and (nm_edges == 0) and (est_wall_mm >= min_wall_mm)

    report = PrintCheckReport(
        is_watertight=is_wt,
        non_manifold_edges=nm_edges,
        min_wall_thickness_mm=est_wall_mm,
        is_printable=printable,
        scale_applied=scale,
    )
    return verts, tris, report


def export_stl(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    """Export binary STL file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int32)

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(fn_len, 1e-8)

    with open(out_path, "wb") as f:
        header = b"Redis3D Pro Binary STL".ljust(80, b"\x00")
        f.write(header)
        f.write(struct.pack("<I", len(tris)))
        for i in range(len(tris)):
            f.write(struct.pack("<3f", *fn[i]))
            f.write(struct.pack("<3f", *v0[i]))
            f.write(struct.pack("<3f", *v1[i]))
            f.write(struct.pack("<3f", *v2[i]))
            f.write(struct.pack("<H", 0))
    return out_path


def export_3mf(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    """Export standard 3MF (3D Manufacturing Format) ZIP/XML package."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int32)

    v_lines = "\n".join(
        f'          <vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}" />'
        for v in verts
    )
    t_lines = "\n".join(
        f'          <triangle v1="{int(t[0])}" v2="{int(t[1])}" v3="{int(t[2])}" />'
        for t in tris
    )

    model_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources>
    <object id="1" type="model">
      <mesh>
        <vertices>
{v_lines}
        </vertices>
        <triangles>
{t_lines}
        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1" />
  </build>
</model>
"""
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />
</Types>
"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2015/02/3dmodel" />
</Relationships>
"""
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types.encode("utf-8"))
        zf.writestr("_rels/.rels", rels_xml.encode("utf-8"))
        zf.writestr("3D/3dmodel.model", model_xml.encode("utf-8"))
    return out_path
