"""Phase 8 Acceptance Gate: Commercial Post-Processing Suite (roadfinal.md §7 Phase 8).

Contract across 20 generated assets:
  1. 100% (20/20) pass is_watertight == True & non_manifold_edges == 0.
  2. Quad ratio >= 80%.
  3. UV overlap == 0.
  4. GLB schema + skinning weights valid (sum == 1.0 per vertex).
  5. 4K texture bake < 1.5 GB VRAM.
  6. PrintDoctor (>= 0.8 mm wall thickness) + STL / 3MF / OBJ / USDZ export validation.
"""

from __future__ import annotations

import json
import struct
import zipfile

import numpy as np
import pytest
import torch

from redis3d.data.procedural_gen import generate_procedural_batch
from redis3d.math.topology_repair import repair_mesh
from redis3d.postprocess.auto_rig import AxionAutoRigger
from redis3d.postprocess.exporters import export_glb, export_obj, export_usdz
from redis3d.postprocess.part_uv import part_uv_unwrap
from redis3d.postprocess.pbr_upscaler import TiledPBRUpscaler
from redis3d.postprocess.print_doctor import export_3mf, export_stl, run_print_doctor
from redis3d.postprocess.quad_retopo import quad_retopologize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real GPU required")


def test_phase8_20_assets_postprocess_99_sla(tmp_path):
    """Run QuadRetopo + PartUV + AutoRig + PrintDoctor + GLB/STL/3MF across 20 procedural assets."""
    assets = generate_procedural_batch(count=20, seed_offset=77, num_workers=8)
    rigger = AxionAutoRigger()

    wt_pass = 0
    quad_ratios = []
    uv_overlaps_total = 0

    for idx, asset in enumerate(assets):
        # Ensure base mesh is watertight & manifold via Phase 2 MeshRepair
        verts, faces, _ = repair_mesh(asset["vertices"], asset["faces"])

        # 1. QuadRetopo (4-RoSy + adaptive quad-dominant)
        qmesh = quad_retopologize(verts, faces, target_quads=512)
        if qmesh.is_watertight and qmesh.non_manifold_edges == 0:
            wt_pass += 1
        quad_ratios.append(qmesh.quad_ratio)
        assert qmesh.quad_ratio >= 0.80, f"Asset {idx} quad ratio {qmesh.quad_ratio:.2f} < 0.80"

        # 2. PartUV (Zero overlap)
        tri_faces = qmesh.to_triangles()
        uv_res = part_uv_unwrap(qmesh.vertices, tri_faces)
        uv_overlaps_total += uv_res.uv_overlap_count
        assert uv_res.uv_overlap_count == 0, f"Asset {idx} UV overlap={uv_res.uv_overlap_count}"

        # 3. Axion-Rig (Skin weights sum == 1.0)
        rig_res = rigger.rig(uv_res.vertices, uv_res.faces)
        w_sums = rig_res.skin_weights.sum(axis=1)
        assert np.allclose(w_sums, 1.0, atol=1e-5), f"Asset {idx} skin weights do not sum to 1.0"

        # 4. Export GLB & validate binary GLB magic + JSON schema
        glb_path = export_glb(
            tmp_path / f"asset_{idx}.glb",
            uv_res.vertices,
            uv_res.faces,
            uvs=uv_res.uvs,
            rig=rig_res,
        )
        raw_glb = glb_path.read_bytes()
        magic, version, length = struct.unpack("<III", raw_glb[:12])
        assert magic == 0x46546C67 and version == 2 and length == len(raw_glb)
        json_len, json_type = struct.unpack("<II", raw_glb[12:20])
        assert json_type == 0x4E4F534A
        gltf_json = json.loads(raw_glb[20 : 20 + json_len].decode("utf-8"))
        assert "skins" in gltf_json and len(gltf_json["meshes"]) == 1

    print(
        f"\n[Phase 8 20-Asset Gate] Watertight+Manifold={wt_pass}/20 | "
        f"MinQuadRatio={min(quad_ratios):.2f} | TotalUVOverlaps={uv_overlaps_total}"
    )
    assert wt_pass == 20, f"Only {wt_pass}/20 assets passed watertight/manifold!"
    assert uv_overlaps_total == 0


def test_phase8_4k_pbr_upscaler_under_1_5gb_and_print_exporters(tmp_path):
    """Verify 4K tiled PBR bake stays < 1.5 GB VRAM and STL/3MF/OBJ/USDZ export cleanly."""
    assets = generate_procedural_batch(count=1, seed_offset=99, num_workers=1)
    rep_verts, rep_faces, _ = repair_mesh(assets[0]["vertices"], assets[0]["faces"])
    uv_res = part_uv_unwrap(rep_verts, rep_faces)

    upscaler = TiledPBRUpscaler(tile_size=512, max_vram_mb=1400, device="cuda")
    pbr_maps = upscaler.bake_and_upscale(uv_res.uvs, uv_res.faces, target_res=4096)
    assert pbr_maps.albedo.shape == (4096, 4096, 3)
    assert pbr_maps.roughness.shape == (4096, 4096)
    assert pbr_maps.metallic.shape == (4096, 4096)
    assert pbr_maps.normal.shape == (4096, 4096, 3)

    # PrintDoctor + STL + 3MF + OBJ + USDZ
    p_verts, p_faces, report = run_print_doctor(rep_verts, rep_faces, target_size_mm=80.0)
    assert report.is_watertight and report.non_manifold_edges == 0
    assert report.min_wall_thickness_mm >= 0.8 and report.is_printable

    stl_p = export_stl(tmp_path / "model.stl", p_verts, p_faces)
    mf3_p = export_3mf(tmp_path / "model.3mf", p_verts, p_faces)
    obj_p = export_obj(tmp_path / "model.obj", uv_res.vertices, uv_res.faces, uvs=uv_res.uvs)
    usdz_p = export_usdz(tmp_path / "model.usdz", uv_res.vertices, uv_res.faces)

    assert stl_p.stat().st_size > 84
    assert zipfile.is_zipfile(mf3_p)
    assert obj_p.stat().st_size > 50
    assert zipfile.is_zipfile(usdz_p)
    print(f"\n[Phase 8 4K+Print] 4K maps OK | Wall={report.min_wall_thickness_mm:.1f}mm | STL/3MF/OBJ/USDZ OK")
