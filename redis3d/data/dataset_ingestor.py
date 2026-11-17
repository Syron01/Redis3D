"""Local 3D asset ingestor and .wov shard compiler (roadfinal.md Phase 3).

Ingests 3D assets (procedural batches or local .glb/.obj/.ply/.stl files),
audits provenance via license_audit, compiles to W-OVoxel^2 octree tokens via
mesh_to_wov_asset, and writes high-density Zstandard .wov shards (< 160 KB/asset).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from redis3d.data.license_audit import audit_asset_license
from redis3d.data.mesh_to_wov import mesh_to_wov_asset, warmup_mesh_to_wov
from redis3d.data.procedural_gen import compute_face_and_vertex_normals, normalize_to_bounds
from redis3d.data.wov_format import save_wov_shard, verify_wov_shard


def load_mesh_file(file_path: Path | str, default_license: str = "UNKNOWN") -> dict[str, Any]:
    """Load an arbitrary 3D mesh file (.glb, .gltf, .obj, .ply, .stl) into an asset dictionary."""
    import trimesh

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"3D mesh file not found: {path}")

    # Load with trimesh
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Unsupported 3D geometry in {path}: {type(loaded)}")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Normalize within [-0.90, 0.90]^3
    vertices = normalize_to_bounds(vertices)

    # Compute normals if missing or invalid
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(vertices):
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    else:
        _, normals = compute_face_and_vertex_normals(vertices, faces)

    # PBR materials or color
    albedo = np.tile(np.array([0.70, 0.70, 0.70], dtype=np.float32), (len(faces), 1))
    roughness = np.full((len(faces), 1), 0.35, dtype=np.float32)
    metallic = np.full((len(faces), 1), 0.0, dtype=np.float32)

    part_ids = np.zeros(len(faces), dtype=np.int64)

    # Inspect license in metadata or companion .json sidecar
    sidecar_json = path.with_suffix(".json")
    license_tag = default_license
    if sidecar_json.is_file():
        try:
            with sidecar_json.open("r", encoding="utf-8") as f:
                meta = json.load(f)
                license_tag = meta.get("license_tag") or meta.get("license", default_license)
        except Exception:
            pass

    return {
        "asset_id": path.stem,
        "category": "imported_mesh",
        "license_tag": license_tag,
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "albedo": albedo,
        "roughness": roughness,
        "metallic": metallic,
        "part_ids": part_ids,
        "open_sheet": False,
        "source_path": str(path.resolve()),
    }


def ingest_corpus(
    assets_or_files: Iterable[dict[str, Any] | Path | str],
    shard_dir: Path | str,
    shard_size: int = 64,
    octree_level: int = 7,
    prune: bool = False,
    device: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Ingest, license-audit, compile, and shard a corpus of 3D assets.

    Args:
        assets_or_files: Iterable of asset dictionaries or file paths.
        shard_dir: Destination directory for .wov shards and corpus manifest.
        shard_size: Number of assets packed per .wov shard (default: 64).
        octree_level: Target W-OVoxel^2 octree resolution level (default: 7).
        prune: Whether to apply 3D Haar-lifting wavelet compression.
        device: 'cuda' or 'cpu' (defaults to 'cuda' if available).
        on_progress: Optional progress callback (current, total).

    Returns:
        List of generated shard file paths.
    """
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    warmup_mesh_to_wov(device=device)

    items = list(assets_or_files)
    total_items = len(items)

    shard_paths: list[Path] = []
    buffer: list[dict[str, Any]] = []
    shard_idx = 0

    accepted_count = 0
    rejected_count = 0
    rejected_assets: list[dict[str, Any]] = []

    for idx, item in enumerate(items):
        # 1. Resolve to asset dictionary
        if isinstance(item, (str, Path)):
            asset = load_mesh_file(item)
        elif isinstance(item, dict):
            asset = item
        else:
            raise TypeError(f"Unsupported asset item type: {type(item)}")

        # 2. Strict commercial license audit
        if not audit_asset_license(asset):
            rejected_count += 1
            rejected_assets.append({
                "asset_id": asset.get("asset_id", "unknown"),
                "license_tag": asset.get("license_tag", "UNKNOWN"),
            })
            continue

        # 3. Compile to W-OVoxel^2 tokens
        compiled = mesh_to_wov_asset(
            asset,
            octree_level=octree_level,
            device=device,
            prune=prune,
        )

        buffer.append(compiled)
        accepted_count += 1

        # 4. Flush shard when buffer is full
        if len(buffer) >= shard_size:
            shard_path = shard_dir / f"shard_{shard_idx:04d}.wov"
            save_wov_shard(buffer, shard_path)
            shard_paths.append(shard_path)
            buffer.clear()
            shard_idx += 1

        if on_progress is not None:
            on_progress(idx + 1, total_items)

    # 5. Flush remaining partial shard
    if buffer:
        shard_path = shard_dir / f"shard_{shard_idx:04d}.wov"
        save_wov_shard(buffer, shard_path)
        shard_paths.append(shard_path)
        buffer.clear()

    # 6. Write corpus manifest
    manifest = {
        "total_items": total_items,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "rejected_assets": rejected_assets,
        "num_shards": len(shard_paths),
        "shard_paths": [p.name for p in shard_paths],
        "shard_size": shard_size,
        "octree_level": octree_level,
        "pruned": prune,
    }

    manifest_path = shard_dir / "corpus_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return shard_paths
