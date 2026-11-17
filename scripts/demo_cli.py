"""Redis3D Pro Production CLI Entry Point (roadfinal.md §8)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from redis3d.math.topology_repair import repair_mesh
from redis3d.models.pipeline import Redis3DProPipeline
from redis3d.postprocess.auto_rig import AxionAutoRigger
from redis3d.postprocess.exporters import export_glb, export_obj, export_usdz
from redis3d.postprocess.part_uv import part_uv_unwrap
from redis3d.postprocess.print_doctor import export_3mf, export_stl, run_print_doctor
from redis3d.postprocess.quad_retopo import quad_retopologize
from redis3d.ui.server import _create_manifold_proxy_mesh


def run_cli(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Redis3D Pro Commercial 3D Generator")
    parser.add_argument("--image", type=str, default=None, help="Input concept image path")
    parser.add_argument("--prompt", type=str, default="precision armored mech helmet with sharp beveled visor")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--patr-steps", type=int, default=6)
    parser.add_argument("--cad-snap", action="store_true")
    parser.add_argument("--symmetry", action="store_true")
    parser.add_argument("--retopo-quads", type=int, default=512)
    parser.add_argument("--auto-rig", action="store_true")
    parser.add_argument("--tex-res", type=int, default=1024)
    parser.add_argument("--print-check", action="store_true")
    parser.add_argument("--vram-mode", choices=["resident", "streaming"], default="resident")
    parser.add_argument("--max-vram-mb", type=int, default=1800)
    parser.add_argument("--out", type=str, default="output.glb")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()

    # Load or synthesize input image tensor [1, 3, 128, 128]
    img_t = torch.rand(1, 3, 128, 128, device=device)
    if args.image and Path(args.image).exists():
        try:
            from PIL import Image

            pil_img = Image.open(args.image).convert("RGB").resize((128, 128))
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        except Exception:
            pass

    pipe = Redis3DProPipeline(vram_mode=args.vram_mode, max_vram_mb=args.max_vram_mb, device=device)
    out = pipe.generate(
        image=img_t,
        prompt=args.prompt,
        num_candidates=args.candidates,
        nfe_steps=args.steps,
        patr_steps=args.patr_steps,
        seq_len=128,
        sym_lock=args.symmetry,
    )

    verts, faces, v_pbr = _create_manifold_proxy_mesh(out.coords, out.pbr)
    verts, faces, _ = repair_mesh(verts, faces)
    qmesh = quad_retopologize(verts, faces, target_quads=args.retopo_quads)
    uv_res = part_uv_unwrap(qmesh.vertices, qmesh.to_triangles())
    rig_res = AxionAutoRigger().rig(uv_res.vertices, uv_res.faces) if args.auto_rig else None

    out_path = Path(args.out)
    ext = out_path.suffix.lower()
    if ext == ".stl":
        pv, pf, rep = run_print_doctor(uv_res.vertices, uv_res.faces)
        export_stl(out_path, pv, pf)
    elif ext == ".3mf":
        pv, pf, rep = run_print_doctor(uv_res.vertices, uv_res.faces)
        export_3mf(out_path, pv, pf)
    elif ext == ".obj":
        export_obj(out_path, uv_res.vertices, uv_res.faces, uvs=uv_res.uvs)
    elif ext == ".usdz":
        export_usdz(out_path, uv_res.vertices, uv_res.faces)
    else:
        export_glb(
            out_path,
            uv_res.vertices,
            uv_res.faces,
            uvs=uv_res.uvs,
            rig=rig_res,
            metallic=float(np.clip(v_pbr[:, 4].mean(), 0.0, 1.0)),
            roughness=float(np.clip(v_pbr[:, 3].mean(), 0.05, 1.0)),
        )

    if args.print_check:
        _, _, rep = run_print_doctor(uv_res.vertices, uv_res.faces)
        print(f"[PrintDoctor] Watertight={rep.is_watertight} | Wall={rep.min_wall_thickness_mm:.2f}mm | Printable={rep.is_printable}")

    dt = time.perf_counter() - t0
    print(f"[Redis3D CLI] Exported {out_path} in {dt:.3f}s")
    return out_path


if __name__ == "__main__":
    run_cli()
