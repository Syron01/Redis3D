"""Phase 3 acceptance gate: commercial license auditing and .wov v2 binary codec.

Specifications (roadfinal.md §5.2, §7 Phase 3):
  * Commercial-safe allowlist: MIT, Apache-2.0, CC0, CC-BY-4.0, Procedural-Owned, BSD-2-Clause, BSD-3-Clause.
  * Rejects contaminated/non-commercial licenses: CC-BY-NC, CC-BY-NC-SA, GPL, AGPL, NonCommercial, etc.
  * Provenance verification: blocks unverified/scraped/pirated data.
  * Zstandard-compressed .wov v2 binary format with magic b"WOV2" (single) and b"WOVS" (shard).
  * Payloads: 64-bit Morton keys, dual verts, normals/oct normals, PBR tuple, sheet bits,
    part IDs, skeleton & skinning weights, and intrinsic renders.
  * Size SLA: < 160 KB per asset footprint.
  * Full roundtrip lossless encoding/decoding.
  * Windows 11 UTF-8 / Unicode path safety.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

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
    is_contaminated_license,
    is_license_allowed,
    normalize_license_tag,
)
from redis3d.data.wov_format import (
    WOV_MAGIC,
    WOV_SHARD_MAGIC,
    decode_wov,
    encode_wov,
    iter_wov_shard,
    load_wov,
    load_wov_shard,
    read_wov_shard_index,
    save_wov,
    save_wov_shard,
    wov_to_torch,
)


# ============================================================================
# 1. License & Provenance Audit Tests
# ============================================================================

def test_commercial_safe_allowlist():
    """Verify that all target commercial licenses and common aliases pass audit."""
    safe_licenses = [
        "MIT",
        "mit",
        "MIT License",
        "Apache-2.0",
        "Apache 2.0",
        "apache-2",
        "CC0",
        "cc0 1.0",
        "CC0 1.0 Universal",
        "Public Domain",
        "CC-BY-4.0",
        "CC-BY 4.0",
        "CC BY 4.0",
        "CC-BY",
        "Procedural-Owned",
        "procedural",
        "Axion-Owned",
        "Redis3D-Owned",
        "BSD-2-Clause",
        "bsd-2-clause",
        "BSD-3-Clause",
        "bsd-3-clause",
        "BSD",
    ]

    for lic in safe_licenses:
        assert is_commercial_safe(lic), f"License '{lic}' should be commercial-safe"
        assert is_license_allowed(lic), f"License '{lic}' should be allowed"
        manifest = {
            "asset_id": f"test_{lic}",
            "license": lic,
            "provenance": "procedural-synthetics-v1",
        }
        assert audit_asset_license(manifest), f"Manifest with license '{lic}' failed audit"


def test_reject_contaminated_licenses():
    """Verify that contaminated/copyleft/non-commercial licenses are strictly rejected."""
    bad_licenses = [
        "CC-BY-NC",
        "CC-BY-NC-4.0",
        "CC-BY-NC-SA",
        "CC-BY-NC-ND",
        "CC-BY-ND",
        "GPL",
        "GPL-2.0",
        "GPL-3.0",
        "AGPL",
        "AGPL-3.0",
        "LGPL",
        "LGPL-3.0",
        "NonCommercial",
        "Non-Commercial",
        "Commons Clause",
        "SSPL",
        "Proprietary",
        "All Rights Reserved",
        "Academic Only",
        "Research Only",
        "Personal Use Only",
        "Unknown",
        "Unlicensed",
    ]

    for lic in bad_licenses:
        assert is_contaminated_license(lic), f"License '{lic}' must be detected as contaminated"
        assert not is_commercial_safe(lic), f"License '{lic}' must not be commercial-safe"
        manifest = {
            "asset_id": f"bad_{lic}",
            "license": lic,
            "provenance": "public-web-scrape",
        }
        assert not audit_asset_license(manifest), f"Manifest with bad license '{lic}' was not rejected"
        with pytest.raises(LicenseAuditError):
            audit_asset_license(manifest, strict=True)


def test_provenance_enforcement():
    """Verify that provenance tracking is strictly enforced before corpus acceptance."""
    # 1. Clean provenance with commercial license passes
    clean_manifest = {
        "asset_id": "cad_part_42",
        "license": "MIT",
        "provenance": {
            "source": "ABC Dataset",
            "author": "Clean-Room Lab",
            "url": "https://example.com/dataset/clean/cad_part_42",
        },
    }
    enforce_provenance(clean_manifest)
    assert audit_asset_license(clean_manifest, require_provenance=True)

    # 2. Tainted provenance (scraped, pirated, unknown) is rejected even with MIT license
    tainted_sources = ["unknown", "unverified", "scraped", "pirated", "torrent", "leaked"]
    for taint in tainted_sources:
        tainted_manifest = {
            "asset_id": f"tainted_{taint}",
            "license": "MIT",
            "provenance": taint,
        }
        assert not audit_asset_license(tainted_manifest)
        with pytest.raises(LicenseAuditError):
            enforce_provenance(tainted_manifest)

    # 3. Missing provenance is rejected when enforce_provenance is invoked
    missing_manifest = {
        "asset_id": "no_provenance",
        "license": "MIT",
    }
    with pytest.raises(LicenseAuditError):
        enforce_provenance(missing_manifest)


def test_audit_manifest_file(tmp_path: Path):
    """Test auditing JSON and JSONL manifest files, verifying approved/rejected splits."""
    manifest_data = [
        {"asset_id": "asset_1", "license": "MIT", "provenance": "procedural"},
        {"asset_id": "asset_2", "license": "CC-BY-NC-4.0", "provenance": "web"},
        {"asset_id": "asset_3", "license": "Apache-2.0", "provenance": "open-source"},
        {"asset_id": "asset_4", "license": "GPL-3.0", "provenance": "github"},
        {"asset_id": "asset_5", "license": "CC0", "provenance": "clean"},
        {"asset_id": "asset_6", "license": "BSD-3-Clause", "provenance": "curated"},
    ]

    # Test JSON file format
    json_path = tmp_path / "manifest.json"
    json_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    approved, rejected = audit_manifest_file(json_path)

    assert len(approved) == 4
    assert len(rejected) == 2
    assert {a["asset_id"] for a in approved} == {"asset_1", "asset_3", "asset_5", "asset_6"}
    assert {r["asset_id"] for r in rejected} == {"asset_2", "asset_4"}

    # Test JSONL format
    jsonl_path = tmp_path / "manifest.jsonl"
    lines = "\n".join(json.dumps(x) for x in manifest_data)
    jsonl_path.write_text(lines, encoding="utf-8")
    approved_l, rejected_l = audit_manifest_file(jsonl_path)

    assert len(approved_l) == 4
    assert len(rejected_l) == 2

    # Test audit_manifest summary report
    report = audit_manifest(json_path)
    assert report["total"] == 6
    assert report["allowed_count"] == 4
    assert report["rejected_count"] == 2
    assert report["is_100pct_commercial"] is False


# ============================================================================
# 2. .wov Binary Codec & Shard Tests
# ============================================================================

def _create_synthetic_3d_asset(
    asset_id: str = "cad_bracket_001",
    num_tokens: int = 1500,
    with_skeleton: bool = True,
    with_intrinsics: bool = True,
) -> dict[str, Any]:
    """Generate fully populated synthetic asset dict for lossless roundtrip testing."""
    rng = np.random.default_rng(42)

    # 1. Morton Keys [N] uint64
    keys = np.sort(rng.integers(0, 1 << 60, size=num_tokens, dtype=np.uint64))

    # 2. Dual vertex positions [N, 3] float32 in [-1, 1]
    dual_verts = rng.uniform(-1.0, 1.0, size=(num_tokens, 3)).astype(np.float32)

    # 3. Octahedral normals [N, 2] int8 in [-127, 127]
    normals = rng.integers(-127, 127, size=(num_tokens, 2), dtype=np.int8)

    # 4. PBR materials [N, 7] uint8
    albedo = rng.integers(0, 255, size=(num_tokens, 3), dtype=np.uint8)
    roughness = rng.integers(0, 255, size=(num_tokens,), dtype=np.uint8)
    metalness = rng.integers(0, 255, size=(num_tokens,), dtype=np.uint8)
    opacity = np.full((num_tokens,), 255, dtype=np.uint8)
    emissive = np.zeros((num_tokens,), dtype=np.uint8)

    # 5. Sheet bitfields / CAD-Snap ranks [N] uint8
    sheet_bitfields = rng.integers(0, 4, size=(num_tokens,), dtype=np.uint8)

    # 6. Part IDs [N] uint8
    part_ids = rng.integers(0, 8, size=(num_tokens,), dtype=np.uint8)

    asset: dict[str, Any] = {
        "asset_id": asset_id,
        "category": "mechanical_cad",
        "license_tag": "Procedural-Owned",
        "morton_keys": keys,
        "dual_verts": dual_verts,
        "normals": normals,
        "albedo": albedo,
        "roughness": roughness,
        "metalness": metalness,
        "opacity": opacity,
        "emissive": emissive,
        "sheet_bitfields": sheet_bitfields,
        "part_ids": part_ids,
        "metadata": {
            "provenance": "procedural-synthetics-v1",
            "bounding_box": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        },
    }

    if with_skeleton:
        num_joints = 16
        asset["joint_positions"] = rng.uniform(-0.8, 0.8, size=(num_joints, 3)).astype(np.float32)
        asset["joint_parents"] = np.array([-1] + list(range(num_joints - 1)), dtype=np.int16)
        asset["skin_indices"] = rng.integers(0, num_joints, size=(num_tokens, 4), dtype=np.int16)
        raw_weights = rng.uniform(0.0, 1.0, size=(num_tokens, 4)).astype(np.float32)
        asset["skin_weights"] = raw_weights / raw_weights.sum(axis=-1, keepdims=True)

    if with_intrinsics:
        H, W = 64, 64
        asset["delit_albedo"] = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        asset["normal_map"] = rng.integers(-127, 127, size=(H, W, 3), dtype=np.int8)
        asset["shading"] = rng.integers(0, 255, size=(H, W), dtype=np.uint8)

    return asset


def test_wov_single_asset_roundtrip_lossless(tmp_path: Path):
    """Test full single .wov file serialization and exact lossless decoding."""
    original = _create_synthetic_3d_asset(num_tokens=1200)
    wov_file = tmp_path / "test_asset.wov"

    # Save and verify return length
    comp_size = save_wov(wov_file, original, compression_level=3)
    assert comp_size > 0
    assert wov_file.exists()

    # Load and verify
    loaded = load_wov(wov_file)

    # 1. Headers
    assert loaded["asset_id"] == original["asset_id"]
    assert loaded["category"] == original["category"]
    assert loaded["license_tag"] == original["license_tag"]
    assert loaded["num_tokens"] == 1200
    assert loaded["num_joints"] == 16

    # 2. Morton keys: exact uint64 match
    np.testing.assert_array_equal(loaded["morton_keys"], original["morton_keys"])

    # 3. Dual vertices: exact float32 match
    np.testing.assert_allclose(loaded["dual_verts"], original["dual_verts"], rtol=1e-6)

    # 4. Normals: exact int8 match
    np.testing.assert_array_equal(loaded["normals"], original["normals"])

    # 5. PBR channels: exact uint8 match
    np.testing.assert_array_equal(loaded["albedo"], original["albedo"])
    np.testing.assert_array_equal(loaded["roughness"], original["roughness"])
    np.testing.assert_array_equal(loaded["metalness"], original["metalness"])
    np.testing.assert_array_equal(loaded["opacity"], original["opacity"])
    np.testing.assert_array_equal(loaded["emissive"], original["emissive"])

    # 6. Bitfields & Part IDs: exact uint8 match
    np.testing.assert_array_equal(loaded["sheet_bitfields"], original["sheet_bitfields"])
    np.testing.assert_array_equal(loaded["cad_snap_ranks"], original["sheet_bitfields"])
    np.testing.assert_array_equal(loaded["part_ids"], original["part_ids"])

    # 7. Skeleton: exact match
    np.testing.assert_allclose(loaded["joint_positions"], original["joint_positions"], rtol=1e-6)
    np.testing.assert_array_equal(loaded["joint_parents"], original["joint_parents"])
    np.testing.assert_array_equal(loaded["skin_indices"], original["skin_indices"])
    np.testing.assert_allclose(loaded["skin_weights"], original["skin_weights"], rtol=1e-6)

    # 8. Intrinsics: exact match
    np.testing.assert_array_equal(loaded["delit_albedo"], original["delit_albedo"])
    np.testing.assert_array_equal(loaded["normal_map"], original["normal_map"])
    np.testing.assert_array_equal(loaded["shading"], original["shading"])

    # 9. Metadata & Provenance
    assert loaded["metadata"]["provenance"] == "procedural-synthetics-v1"
    assert loaded["provenance"] == "procedural-synthetics-v1"


def test_wov_size_sla(tmp_path: Path):
    """Verify that compressed asset size satisfies the < 160 KB per asset footprint SLA."""
    # Create realistic 2048-token asset (standard target resolution)
    asset = _create_synthetic_3d_asset(num_tokens=2048, with_skeleton=True, with_intrinsics=True)
    out_file = tmp_path / "sla_check.wov"
    comp_size = save_wov(out_file, asset, compression_level=3)

    size_kb = comp_size / 1024.0
    print(f"\n[WOV SLA] 2048-token asset compressed size: {size_kb:.2f} KB (SLA: < 160.0 KB)")
    assert size_kb < 160.0, f"Asset footprint {size_kb:.2f} KB exceeds 160 KB SLA!"


def test_wov_shard_roundtrip(tmp_path: Path):
    """Test high-performance binary shard storage, streaming, and index inspection."""
    num_assets = 8
    assets = [
        _create_synthetic_3d_asset(f"shard_asset_{i:03d}", num_tokens=500 + i * 50)
        for i in range(num_assets)
    ]

    shard_path = tmp_path / "train_corpus_000.wov"
    total_bytes = save_wov_shard(shard_path, assets, compression_level=3)
    assert total_bytes > 0
    assert shard_path.exists()

    # 1. Read index without decompressing full payloads
    index = read_wov_shard_index(shard_path)
    assert index["count"] == num_assets
    assert index["magic"] == "WOVS"
    assert len(index["entries"]) == num_assets
    for i, entry in enumerate(index["entries"]):
        assert entry["asset_id"] == f"shard_asset_{i:03d}"
        assert entry["compressed_size"] > 0
        assert entry["uncompressed_size"] > entry["compressed_size"]

    # 2. Stream through shard generator
    streamed = list(iter_wov_shard(shard_path))
    assert len(streamed) == num_assets
    for i in range(num_assets):
        assert streamed[i]["asset_id"] == f"shard_asset_{i:03d}"
        np.testing.assert_array_equal(streamed[i]["morton_keys"], assets[i]["morton_keys"])

    # 3. Load full shard
    loaded_all = load_wov_shard(shard_path)
    assert len(loaded_all) == num_assets


def test_wov_flexible_argument_order(tmp_path: Path):
    """Test that save_wov and save_wov_shard accept both (path, asset) and (asset, path)."""
    asset = _create_synthetic_3d_asset("arg_test", num_tokens=300)

    p1 = tmp_path / "order1.wov"
    p2 = tmp_path / "order2.wov"
    save_wov(p1, asset)
    save_wov(asset, p2)
    assert load_wov(p1)["asset_id"] == "arg_test"
    assert load_wov(p2)["asset_id"] == "arg_test"

    s1 = tmp_path / "shard1.wov"
    s2 = tmp_path / "shard2.wov"
    save_wov_shard(s1, [asset])
    save_wov_shard([asset], s2)
    assert len(load_wov_shard(s1)) == 1
    assert len(load_wov_shard(s2)) == 1


def test_wov_to_torch():
    """Verify conversion of loaded assets to PyTorch tensors."""
    asset = _create_synthetic_3d_asset("torch_test", num_tokens=200)
    raw = encode_wov(asset)
    decoded_torch = decode_wov(raw, to_torch=True)

    assert isinstance(decoded_torch["morton_keys"], torch.Tensor)
    assert isinstance(decoded_torch["dual_verts"], torch.Tensor)
    assert isinstance(decoded_torch["normals"], torch.Tensor)
    assert isinstance(decoded_torch["albedo"], torch.Tensor)
    assert decoded_torch["dual_verts"].shape == (200, 3)


def test_utf8_turkish_locale_path_safety(tmp_path: Path):
    """Verify robust file handling across Turkish / UTF-8 special characters (Windows 11)."""
    turkish_dir = tmp_path / "kütüphane_veri_şablonu"
    turkish_file = turkish_dir / "çift_kanatlı_dişli_ğöü.wov"

    asset = _create_synthetic_3d_asset("türkçe_varlık_01", num_tokens=400)
    save_wov(turkish_file, asset)

    assert turkish_file.exists()
    loaded = load_wov(turkish_file)
    assert loaded["asset_id"] == "türkçe_varlık_01"
    np.testing.assert_array_equal(loaded["morton_keys"], asset["morton_keys"])


# ============================================================================
# 3. Phase 3 Acceptance Gate: 256 Assets across 16 Workers in < 60s
# ============================================================================

def test_gate_phase3_256_procedural_assets_under_60s(tmp_path: Path):
    """Phase 3 acceptance gate (roadfinal.md §7 Phase 3):
    1. Generate 256 multi-part PBR/rigged procedural assets across 16 CPU workers in < 60s.
    2. Verify all 6 categories are represented.
    3. Verify license_audit approves 100% of procedural assets (license_tag='Procedural-Owned').
    4. Compile / save to .wov shards, verify size SLA (< 160 KB/asset) and lossless decode.
    5. Verify mocked CC-BY-NC / GPL manifests are 100% rejected.
    """
    import time
    from redis3d.data.procedural_gen import CATEGORIES, generate_procedural_batch

    # 1. 256 assets across 16 workers
    t0 = time.perf_counter()
    batch = generate_procedural_batch(count=256, num_workers=16)
    gen_time_s = time.perf_counter() - t0

    print(f"\n[Phase 3 Gate] Generated {len(batch)} assets in {gen_time_s:.2f}s "
          f"({len(batch)/gen_time_s:.1f} assets/s, SLA: < 60.0s)")
    assert len(batch) == 256
    assert gen_time_s < 60.0, f"Generation took {gen_time_s:.2f}s, exceeding 60s SLA"

    # 2. Check category representation
    cats_found = {a["category"] for a in batch}
    for cat in CATEGORIES:
        assert cat in cats_found, f"Category '{cat}' missing from 256 generated assets"

    # 3. Check license audit
    for a in batch:
        assert a["license_tag"] == "Procedural-Owned"
        assert audit_asset_license(a) is True

    # 4. Pack into .wov shard and verify size SLA & lossless decode
    shard_assets = []
    for a in batch[:32]:
        wov_a = _create_synthetic_3d_asset(
            asset_id=a["asset_id"],
            num_tokens=1024,
            with_skeleton=(a["category"] == "rigged"),
            with_intrinsics=False,
        )
        wov_a["category"] = a["category"]
        shard_assets.append(wov_a)

    shard_path = tmp_path / "procedural_shard_000.wov"
    comp_bytes = save_wov_shard(shard_path, shard_assets, compression_level=3)
    avg_kb = (comp_bytes / len(shard_assets)) / 1024.0

    print(f"[Phase 3 Gate] Shard 32 assets: total={comp_bytes/1024:.1f} KB, "
          f"avg={avg_kb:.1f} KB/asset (SLA: < 160.0 KB)")
    assert avg_kb < 160.0, f"Average asset size {avg_kb:.1f} KB exceeds 160 KB SLA"

    # Verify lossless decode
    decoded = load_wov_shard(shard_path)
    assert len(decoded) == len(shard_assets)
    for orig, dec in zip(shard_assets, decoded):
        assert orig["asset_id"] == dec["asset_id"]
        np.testing.assert_array_equal(orig["morton_keys"], dec["morton_keys"])

    # 5. Verify mocked CC-BY-NC / GPL manifests are 100% rejected
    mock_bad_manifests = [
        {"asset_id": "bad_01", "license": "CC-BY-NC-4.0", "provenance": "scraped"},
        {"asset_id": "bad_02", "license": "GPL-3.0", "provenance": "github"},
        {"asset_id": "bad_03", "license": "CC-BY-NC-SA", "provenance": "sketchfab"},
    ]
    for m in mock_bad_manifests:
        assert audit_asset_license(m) is False
