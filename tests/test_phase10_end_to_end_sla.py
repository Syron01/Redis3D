"""Phase 10 Acceptance Gate: End-to-End SLA Proof, Tier Configs, CLI & Benchmark Report (roadfinal.md §7 Phase 10)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch
import yaml

from scripts.benchmark_sla import run_benchmark
from scripts.demo_cli import run_cli
from scripts.generate_startup_corpus import build_corpus

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real GPU required")


def test_phase10_tier_configs_valid():
    """Verify tier1, tier2, tier3 YAML configs exist and parse cleanly."""
    for name in ("tier1_nano_5060.yaml", "tier2_studio_5060.yaml", "tier3_cluster_scale.yaml"):
        cfg_path = Path("configs") / name
        assert cfg_path.exists(), f"Missing {cfg_path}"
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert data["model"]["latent_dim"] == 24
        assert data["quantization"]["biloco_rank"] == 256


def test_phase10_startup_corpus_and_demo_cli(tmp_path):
    """Verify startup .wov corpus builder and demo_cli.py end-to-end GLB generation."""
    corpus_stats = build_corpus(out_dir=tmp_path / "shards", num_assets=16, shard_size=8, workers=4)
    assert corpus_stats["assets"] == 16
    assert corpus_stats["shards"] == 2

    out_glb = tmp_path / "mech_helmet.glb"
    res_path = run_cli(
        [
            "--prompt",
            "precision armored mech helmet with sharp beveled visor",
            "--candidates",
            "4",
            "--steps",
            "2",
            "--patr-steps",
            "6",
            "--cad-snap",
            "--symmetry",
            "--retopo-quads",
            "512",
            "--auto-rig",
            "--print-check",
            "--max-vram-mb",
            "1800",
            "--out",
            str(out_glb),
        ]
    )
    assert res_path.exists()
    raw = res_path.read_bytes()
    magic, version, length = struct.unpack("<III", raw[:12])
    assert magic == 0x46546C67 and version == 2 and length == len(raw)


def test_phase10_benchmark_sla_report():
    """Run SLA benchmark and generate BENCHMARK_SLA_REPORT.md proving all SLAs pass."""
    res = run_benchmark(out_report="BENCHMARK_SLA_REPORT.md", num_topo_assets=20)
    assert Path("BENCHMARK_SLA_REPORT.md").exists()
    assert res["resident_vram_mb"] <= 2048.0
    assert res["streaming_vram_mb"] <= 2048.0
    assert res["resident_lat_s"] <= 3.0
    assert res["topo_pct"] >= 99.0
    assert res["uv_pct"] >= 99.0
    assert res["quad_min"] >= 0.80
    print(f"\n[Phase 10 SLA Proof] {res}")
