"""Blender 4.2+ Live Bridge Add-on for Redis3D Pro (roadfinal.md §7 Phase 9)."""

from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

bl_info = {
    "name": "Redis3D Pro Live Studio Bridge",
    "author": "Redis3D Pro",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Redis3D",
    "description": "Generate watertight, rigged PBR GLB assets directly into Blender",
    "category": "3D View",
}


def fetch_glb_from_server(
    prompt: str,
    server_url: str = "http://127.0.0.1:7860",
    api_key: str = "redis3d-local-dev",
    retopo_quads: int = 4000,
    auto_rig: bool = True,
) -> Path:
    """Request a GLB from the local Redis3D Pro server and save to a temp file."""
    payload = json.dumps(
        {
            "prompt": prompt,
            "retopo_quads": retopo_quads,
            "auto_rig": auto_rig,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/api/v1/generate",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        glb_bytes = resp.read()
    out_file = Path(tempfile.gettempdir()) / "redis3d_blender_import.glb"
    out_file.write_bytes(glb_bytes)
    return out_file


try:
    import bpy

    class REDIS3D_OT_Generate(bpy.types.Operator):
        bl_idname = "redis3d.generate_asset"
        bl_label = "Generate 3D Asset"
        bl_options = {"REGISTER", "UNDO"}

        def execute(self, context):
            scene = context.scene
            glb_path = fetch_glb_from_server(
                prompt=getattr(scene, "redis3d_prompt", "armored mech helmet"),
                server_url=getattr(scene, "redis3d_url", "http://127.0.0.1:7860"),
            )
            bpy.ops.import_scene.gltf(filepath=str(glb_path))
            return {"FINISHED"}

    class REDIS3D_PT_Panel(bpy.types.Panel):
        bl_label = "Redis3D Pro"
        bl_idname = "REDIS3D_PT_Panel"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "Redis3D"

        def draw(self, context):
            layout = self.layout
            layout.prop(context.scene, "redis3d_prompt")
            layout.operator("redis3d.generate_asset")

    def register():
        bpy.types.Scene.redis3d_prompt = bpy.props.StringProperty(
            name="Prompt", default="precision armored mech helmet"
        )
        bpy.types.Scene.redis3d_url = bpy.props.StringProperty(
            name="Server URL", default="http://127.0.0.1:7860"
        )
        bpy.utils.register_class(REDIS3D_OT_Generate)
        bpy.utils.register_class(REDIS3D_PT_Panel)

    def unregister():
        bpy.utils.unregister_class(REDIS3D_PT_Panel)
        bpy.utils.unregister_class(REDIS3D_OT_Generate)

except ImportError:
    # Outside Blender Python runtime, keep fetch_glb_from_server & bl_info importable
    pass
