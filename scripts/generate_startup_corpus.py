"""Generate local commercial-clean .wov v2 corpus (roadfinal.md §7 Phase 10)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from redis3d.data.license_audit import audit_manifest
from redis3d.data.mesh_to_wov import mesh_to_wov_asset
from redis3d.data.procedural_gen import generate_procedural_batch
from redis3d.data.wov_format import save_wov_shard


def build_corpus(
    out_dir: str | Path = "data/tier1_wov_shards",
    num_assets: int = 256,
    shard_size: int = 64,
    workers: int = 16,
) -> dict[str, float | int]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    raw_assets = generate_procedural_batch(count=num_assets, num_workers=workers)

    # License audit gate
    manifests = [
        {"asset_id": a.get("asset_id", f"proc_{i}"), "license": "Procedural-Owned"}
        for i, a in enumerate(raw_assets)
    ]
    audit = audit_manifest(manifests)
    assert audit["rejected_count"] == 0, "Unexpected license rejection in procedural corpus!"

    wov_records = [mesh_to_wov_asset(a) for a in raw_assets]
    shard_count = 0
    for start in range(0, len(wov_records), shard_size):
        chunk = wov_records[start : start + shard_size]
        save_wov_shard(out_path / f"shard_{shard_count:04d}.wov", chunk)
        shard_count += 1

    dt = time.perf_counter() - t0
    return {"assets": len(wov_records), "shards": shard_count, "elapsed_s": round(dt, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Redis3D .wov v2 startup corpus")
    parser.add_argument("--out-dir", default="data/tier1_wov_shards")
    parser.add_argument("--num-assets", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    stats = build_corpus(args.out_dir, args.num_assets, args.shard_size, args.workers)
    print(f"[Corpus] Generated {stats['assets']} assets across {stats['shards']} shards in {stats['elapsed_s']}s")


if __name__ == "__main__":
    main()
