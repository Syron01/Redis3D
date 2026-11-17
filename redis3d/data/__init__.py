"""Redis3D data subsystem: license auditing, .wov v2 binary codec, procedural generation, and dataset pipelines."""

from __future__ import annotations

from redis3d.data.dataset_ingestor import (
    ingest_corpus,
    load_mesh_file,
)
from redis3d.data.license_audit import (
    COMMERCIAL_ALLOWLIST,
    CONTAMINATED_PATTERNS,
    LicenseAuditError,
    audit_asset_license,
    audit_manifest,
    audit_manifest_file,
    enforce_provenance,
    filter_allowed_assets,
    is_commercial_safe,
    is_license_allowed,
    normalize_license_tag,
)
from redis3d.data.mesh_to_wov import (
    mesh_to_wov_asset,
    warmup_mesh_to_wov,
)
from redis3d.data.procedural_gen import (
    CATEGORIES,
    generate_procedural_asset,
    generate_procedural_batch,
)
from redis3d.data.wov_format import (
    WOV_MAGIC,
    WOV_SHARD_MAGIC,
    WOV_SHARD_VERSION,
    WOV_VERSION,
    decode_wov,
    encode_wov,
    iter_wov_shard,
    load_wov,
    load_wov_shard,
    read_wov_shard_index,
    save_wov,
    save_wov_shard,
    verify_wov_shard,
    wov_to_torch,
)

__all__ = [
    # License Audit
    "COMMERCIAL_ALLOWLIST",
    "CONTAMINATED_PATTERNS",
    "LicenseAuditError",
    "audit_asset_license",
    "audit_manifest",
    "audit_manifest_file",
    "enforce_provenance",
    "filter_allowed_assets",
    "is_commercial_safe",
    "is_license_allowed",
    "normalize_license_tag",
    # WOV Format
    "WOV_MAGIC",
    "WOV_VERSION",
    "WOV_SHARD_MAGIC",
    "WOV_SHARD_VERSION",
    "encode_wov",
    "decode_wov",
    "save_wov",
    "load_wov",
    "save_wov_shard",
    "load_wov_shard",
    "iter_wov_shard",
    "read_wov_shard_index",
    "verify_wov_shard",
    "wov_to_torch",
    # Procedural Gen
    "CATEGORIES",
    "generate_procedural_asset",
    "generate_procedural_batch",
    # Mesh to WOV
    "mesh_to_wov_asset",
    "warmup_mesh_to_wov",
    # Dataset Ingestor
    "ingest_corpus",
    "load_mesh_file",
]
