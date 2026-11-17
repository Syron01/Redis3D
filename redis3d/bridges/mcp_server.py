"""Local MCP (Model Context Protocol) Tool Server for Redis3D Pro (roadfinal.md §7 Phase 9)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from redis3d.math.topology_repair import mesh_health, repair_mesh
from redis3d.postprocess.print_doctor import run_print_doctor

MCP_TOOLS = [
    {
        "name": "generate_3d_asset",
        "description": "Generate a watertight, rigged 3D GLB asset from a text prompt using Redis3D Pro.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "out_path": {"type": "string"},
                "auto_rig": {"type": "boolean"},
            },
            "required": ["prompt", "out_path"],
        },
    },
    {
        "name": "repair_mesh_topology",
        "description": "Repair a triangle mesh for watertightness and 0 non-manifold edges.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "vertices": {"type": "array"},
                "faces": {"type": "array"},
            },
            "required": ["vertices", "faces"],
        },
    },
    {
        "name": "check_print_readiness",
        "description": "Run PrintDoctor wall-thickness (>= 0.8 mm) and manifold checks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "vertices": {"type": "array"},
                "faces": {"type": "array"},
                "target_size_mm": {"type": "number"},
            },
            "required": ["vertices", "faces"],
        },
    },
]


def handle_mcp_request(req: dict[str, Any]) -> dict[str, Any]:
    """Handle a single JSON-RPC 2.0 MCP request."""
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params", {}) or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "redis3d-pro-mcp", "version": "1.0.0"},
                "capabilities": {"tools": {}},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS}}

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {}) or {}

        if name == "repair_mesh_topology":
            v = np.asarray(args["vertices"], dtype=np.float32)
            f = np.asarray(args["faces"], dtype=np.int32)
            rv, rf, rep = repair_mesh(v, f)
            health = mesh_health(rv, rf)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "vertices_out": int(len(rv)),
                                    "faces_out": int(len(rf)),
                                    "is_watertight": bool(health["is_watertight"]),
                                    "non_manifold_edges": int(health["non_manifold_edges"]),
                                }
                            ),
                        }
                    ]
                },
            }

        if name == "check_print_readiness":
            v = np.asarray(args["vertices"], dtype=np.float32)
            f = np.asarray(args["faces"], dtype=np.int32)
            _, _, report = run_print_doctor(
                v, f, target_size_mm=float(args.get("target_size_mm", 80.0))
            )
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "is_watertight": report.is_watertight,
                                    "is_printable": report.is_printable,
                                    "min_wall_thickness_mm": round(report.min_wall_thickness_mm, 3),
                                }
                            ),
                        }
                    ]
                },
            }

        if name == "generate_3d_asset":
            from redis3d.models.pipeline import Redis3DProPipeline
            from redis3d.postprocess.auto_rig import AxionAutoRigger
            from redis3d.postprocess.exporters import export_glb
            from redis3d.postprocess.part_uv import part_uv_unwrap
            from redis3d.postprocess.quad_retopo import quad_retopologize
            from redis3d.ui.server import _create_manifold_proxy_mesh

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = Redis3DProPipeline(vram_mode="resident", max_vram_mb=2048, device=dev)
            out = pipe.generate(torch.rand(1, 3, 128, 128, device=dev), prompt=args["prompt"], seq_len=128)
            verts, faces, _ = _create_manifold_proxy_mesh(out.coords, out.pbr)
            verts, faces, _ = repair_mesh(verts, faces)
            qmesh = quad_retopologize(verts, faces)
            uv_res = part_uv_unwrap(qmesh.vertices, qmesh.to_triangles())
            rig_res = AxionAutoRigger().rig(uv_res.vertices, uv_res.faces) if args.get("auto_rig", True) else None
            out_p = export_glb(Path(args["out_path"]), uv_res.vertices, uv_res.faces, uvs=uv_res.uvs, rig=rig_res)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps({"glb_path": str(out_p)})}]},
            }

        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool {name}"}}

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method {method}"}}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        resp = handle_mcp_request(req)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
