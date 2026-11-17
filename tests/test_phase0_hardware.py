"""Phase 0 acceptance gate — runs on the REAL RTX 5060 Laptop (sm_120).

Contract (roadfinal.md §7 Phase 0):
  * sm_120 detected, BF16 + FP8 ops verified on-device,
  * AC power / throttle watchdog,
  * VRAMBudgetGuard passes at 1.5 GB, throws at 2.1 GB,
  * 0 MB shared-RAM spill, UTF-8/Turkish path safety.
"""

from __future__ import annotations

import gc

import pytest
import torch

from redis3d.core.hardware_guard import (
    HardwareReport,
    VRAMBudgetExceeded,
    VRAMBudgetGuard,
    power_warnings,
    probe_hardware,
)
from redis3d.core.telemetry import (
    LatencyBreakdown,
    nvml_process_vram_mb,
    vram_snapshot,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Phase 0 gate requires the real NVIDIA GPU"
)

_MB = 1024 * 1024


@pytest.fixture(autouse=True)
def _clean_vram():
    yield
    # exception tracebacks from pytest.raises keep frames (and their CUDA
    # tensors) alive in reference cycles; collect before the next guard
    # measures its baseline, or peaks leak across tests
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def hw() -> HardwareReport:
    return probe_hardware(require_sm120=True)


def test_sm120_probe(hw: HardwareReport):
    assert hw.is_target_sm120, f"expected sm_120, got {hw.capability}"
    assert "5060" in hw.gpu_name
    assert hw.bf16_ok
    assert hw.fp8_scaled_mm_ok, f"FP8 scaled_mm failed: {hw.fp8_detail}"
    assert hw.total_vram_mb >= 7000
    print("\n" + hw.summary())


def test_power_watchdog(hw: HardwareReport):
    assert hw.power["enforced_w"] is not None and hw.power["enforced_w"] > 0
    assert hw.ac_online is not None  # Windows AC status readable
    # Pure warning logic: battery + 50 W throttle must warn, AC + 115 W must not.
    assert len(power_warnings(False, 50.0)) == 2
    assert power_warnings(True, 115.0) == []
    assert len(power_warnings(True, 50.0)) == 1
    print(f"\nAC online={hw.ac_online}, enforced={hw.power['enforced_w']} W, warnings={hw.warnings}")


def test_bf16_matmul_accuracy():
    torch.manual_seed(0)
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    out = (a.bfloat16() @ b.bfloat16()).float()
    ref = a @ b
    rel = ((out - ref).norm() / ref.norm()).item()
    assert torch.isfinite(out).all()
    assert rel < 0.02, f"BF16 relative error too high: {rel}"


def test_fp8_scaled_mm_accuracy():
    torch.manual_seed(0)
    x = torch.randn(256, 256, device="cuda")
    w = torch.randn(256, 256, device="cuda")
    one = torch.ones(1, device="cuda")
    y = torch._scaled_mm(
        x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn).t(),
        scale_a=one, scale_b=one, out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    ref = x @ w.t()
    rel = ((y.float() - ref).norm() / ref.norm()).item()
    assert torch.isfinite(y.float()).all()
    assert rel < 0.10, f"FP8 E4M3 relative error too high: {rel}"
    print(f"\nFP8 E4M3 scaled_mm rel_err={rel:.4f}")


def test_guard_passes_at_1500_mb():
    nbytes = 1536 * _MB
    with VRAMBudgetGuard(max_mb=2048) as guard:
        guard.assert_can_allocate(1536)
        x = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")
        x.fill_(1.0)  # touch every page: real physical VRAM, not lazy mapping
        guard.check()
        assert x.sum().item() == pytest.approx(nbytes // 4, rel=1e-6)
    assert 1500 <= guard.peak_mb <= 2048, f"peak={guard.peak_mb}"
    assert guard.shared_spill_mb == 0.0, f"shared spill={guard.shared_spill_mb} MB"
    print(f"\n1.5 GB pass: peak={guard.peak_mb:.1f} MB, shared_spill={guard.shared_spill_mb} MB")


def test_guard_preflight_blocks_2100_mb():
    with VRAMBudgetGuard(max_mb=2048) as guard:
        with pytest.raises(VRAMBudgetExceeded):
            guard.assert_can_allocate(2100)  # must block BEFORE any allocation/spill
    assert guard.peak_mb < 100  # nothing big was ever allocated


def test_guard_throws_at_2100_mb():
    nbytes = 2100 * _MB
    with pytest.raises(VRAMBudgetExceeded) as ei:
        with VRAMBudgetGuard(max_mb=2048) as guard:
            x = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")
            x.fill_(1.0)
            guard.check()
    assert ei.value.peak_mb > 2048
    print(f"\n2.1 GB correctly rejected: {ei.value}")


def test_nvml_process_tracking():
    x = torch.empty(512 * _MB // 4, dtype=torch.float32, device="cuda")
    x.fill_(1.0)
    torch.cuda.synchronize()
    val = nvml_process_vram_mb()
    # WDDM may hide per-process memory (None); when visible it must be real.
    if val is not None:
        assert val >= 400, f"NVML process VRAM implausible: {val} MB"
    print(f"\nNVML process VRAM: {val} MB (None = WDDM hides it; torch peak is authoritative)")
    del x


def test_vram_snapshot_real():
    snap = vram_snapshot(include_shared=True)
    assert snap.nvml_total_mb >= 7000
    assert snap.nvml_free_mb > 0
    assert snap.shared_spill_mb is not None and snap.shared_spill_mb >= 0
    assert snap.torch_reserved_mb >= snap.torch_allocated_mb
    print(f"\nVRAM snapshot: {snap.as_dict()}")


def test_latency_breakdown_real():
    lat = LatencyBreakdown()
    a = torch.randn(1024, 1024, device="cuda")
    with lat.stage("matmul_work"):
        for _ in range(20):
            a = a @ a
            a = a / a.norm()
    assert lat.stages_ms["matmul_work"] > 0
    assert lat.total_ms >= lat.stages_ms["matmul_work"]
    print(f"\nLatency report: {lat.report()}")


def test_utf8_turkish_path_safety(tmp_path):
    from pathlib import Path

    p = Path(tmp_path) / "Türkçe_ışıl_güç_şirket_ÖÇĞİŞÜ.glb"
    payload = "Redis3D — Türkçe karakter testi: çğıöşü İI ığü"
    p.write_text(payload, encoding="utf-8")
    assert p.read_text(encoding="utf-8") == payload