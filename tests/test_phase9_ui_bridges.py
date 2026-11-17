"""Phase 9 Acceptance Gate: Enterprise Web Studio, API, Blender & MCP Bridges (roadfinal.md §7 Phase 9).

Contract:
  1. Unauthorized API key rejected (401); valid API key accepted.
  2. POST /api/v1/generate returns a valid rigged/retopologized .glb in < 3.5s
     total HTTP turnaround with peak VRAM <= 2.0 GB.
  3. WebSocket /ws/telemetry streams valid VRAM & shared spill JSON.
  4. GET / serves Three.js Studio UI HTML.
  5. Blender add-on metadata + MCP JSON-RPC tools/list & tools/call validated.
"""

from __future__ import annotations

import json
import struct
import time

import pytest
import torch
from fastapi.testclient import TestClient

from redis3d.bridges.blender_addon import bl_info
from redis3d.bridges.mcp_server import handle_mcp_request
from redis3d.ui.server import create_app

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real GPU required")


def test_phase9_api_studio_websocket_and_glb_under_3_5s():
    """Verify end-to-end HTTP GLB generation < 3.5s, <= 2.0 GB VRAM, auth, and WebSocket."""
    app = create_app(vram_mode="resident")
    client = TestClient(app)

    # 1. Studio UI HTML
    r_ui = client.get("/")
    assert r_ui.status_code == 200
    assert "Redis3D Pro Studio" in r_ui.text

    # 2. Health endpoint
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    # 3. API Key Auth Rejection
    r_unauth = client.post("/api/v1/generate", json={"prompt": "test"})
    assert r_unauth.status_code == 401

    # 4. Warmup call + Timed SLA call (< 3.5s)
    payload = {
        "prompt": "precision armored mech helmet with sharp beveled visor",
        "candidates": 4,
        "nfe_steps": 1,
        "patr_steps": 6,
        "retopo_quads": 256,
        "auto_rig": True,
        "sym_lock": True,
    }
    headers = {"X-API-Key": "redis3d-local-dev"}
    _ = client.post("/api/v1/generate", json=payload, headers=headers)

    t0 = time.perf_counter()
    r_gen = client.post("/api/v1/generate", json=payload, headers=headers)
    dt_s = time.perf_counter() - t0

    assert r_gen.status_code == 200
    glb_bytes = r_gen.content
    magic, version, length = struct.unpack("<III", glb_bytes[:12])
    assert magic == 0x46546C67 and version == 2 and length == len(glb_bytes)
    json_len, json_type = struct.unpack("<II", glb_bytes[12:20])
    assert json_type == 0x4E4F534A
    gltf_json = json.loads(glb_bytes[20 : 20 + json_len].decode("utf-8"))
    assert "skins" in gltf_json and len(gltf_json["meshes"]) == 1

    peak_vram_mb = float(r_gen.headers.get("X-Peak-VRAM-MB", "0"))
    print(f"\n[Phase 9 HTTP Turnaround] Time={dt_s:.3f}s (SLA < 3.5s) | Peak VRAM={peak_vram_mb:.1f} MB (SLA <= 2048 MB)")
    assert dt_s < 3.5, f"HTTP turnaround {dt_s:.3f}s >= 3.5s SLA!"
    assert peak_vram_mb <= 2048.0, f"Peak VRAM {peak_vram_mb:.1f} MB > 2048 MB SLA!"

    # 5. WebSocket telemetry
    with client.websocket_connect("/ws/telemetry") as ws:
        msg = ws.receive_json()
        assert "vram_mb" in msg and "shared_spill_mb" in msg


def test_phase9_blender_and_mcp_bridges(tmp_path):
    """Verify Blender 4.2+ addon metadata and MCP JSON-RPC server tools."""
    assert bl_info["blender"] >= (4, 2, 0)

    init_resp = handle_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp["result"]["serverInfo"]["name"] == "redis3d-pro-mcp"

    list_resp = handle_mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tool_names = {t["name"] for t in list_resp["result"]["tools"]}
    assert {"generate_3d_asset", "repair_mesh_topology", "check_print_readiness"} <= tool_names

    out_glb = tmp_path / "mcp_asset.glb"
    call_resp = handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "generate_3d_asset",
                "arguments": {"prompt": "titanium gear", "out_path": str(out_glb), "auto_rig": True},
            },
        }
    )
    assert out_glb.exists() and out_glb.stat().st_size > 100
    assert "glb_path" in call_resp["result"]["content"][0]["text"]
