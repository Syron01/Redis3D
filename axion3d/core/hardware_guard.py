"""Phase 0 guardrails: sm_120 probe, AC-power watchdog, hard VRAM budget.

Enforces the roadfinal.md contract:
  * dense-only sm_120 (Blackwell GB206) execution,
  * AC power / throttle warning via NVML enforced limit + Windows AC status,
  * peak inference VRAM <= 2048 MB with 0 MB shared-RAM spill.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field
from typing import Optional

import pynvml
import torch

from axion3d.core.telemetry import (
    _MB,
    _nvml_handle,
    nvml_process_vram_mb,
    shared_gpu_spill_mb,
)

TARGET_CAPABILITY = (12, 0)  # RTX 5060 Laptop, Blackwell GB206
MIN_AC_POWER_W = 70.0


class HardwareError(RuntimeError):
    pass


class VRAMBudgetExceeded(RuntimeError):
    def __init__(self, peak_mb: float, max_mb: float, source: str):
        self.peak_mb, self.max_mb, self.source = peak_mb, max_mb, source
        super().__init__(
            f"VRAM budget exceeded ({source}): {peak_mb:.1f} MB > {max_mb:.1f} MB hard cap"
        )


class SharedSpillDetected(RuntimeError):
    def __init__(self, spill_mb: float):
        self.spill_mb = spill_mb
        super().__init__(
            f"Shared GPU memory spill detected: {spill_mb:.2f} MB of system RAM "
            "borrowed by the GPU (contract requires 0 MB)"
        )


def _windows_ac_status() -> tuple[Optional[bool], Optional[int]]:
    """(ac_online, battery_percent) from kernel32; (None, None) off-Windows."""
    if os.name != "nt":
        return None, None

    class SYSTEM_POWER_STATUS(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_byte),
            ("BatteryFlag", ctypes.c_byte),
            ("BatteryLifePercent", ctypes.c_byte),
            ("SystemStatusFlag", ctypes.c_byte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = SYSTEM_POWER_STATUS()
    ok = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
    if not ok:
        return None, None
    ac = None if status.ACLineStatus == 255 else status.ACLineStatus == 1
    pct = None if status.BatteryLifePercent == 255 else int(status.BatteryLifePercent)
    return ac, pct


def _power_limits_w(handle) -> dict:
    """NVML power info with laptop-safe fallbacks (GetPowerManagementLimit is
    NotSupported on many GeForce laptops; enforced limit is the truth)."""
    info: dict = {"management_w": None, "enforced_w": None, "default_w": None, "draw_w": None}
    try:
        info["management_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
    except pynvml.NVMLError:
        pass
    try:
        info["enforced_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
    except pynvml.NVMLError:
        pass
    try:
        info["default_w"] = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000.0
    except pynvml.NVMLError:
        pass
    try:
        info["draw_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except pynvml.NVMLError:
        pass
    return info


def power_warnings(ac_online: Optional[bool], enforced_w: Optional[float]) -> list[str]:
    warns: list[str] = []
    if ac_online is False:
        warns.append(
            "Running on BATTERY: expect ~50 W throttle instead of 100-115 W AC "
            "performance. Plug in the adapter before customer demos."
        )
    if enforced_w is not None and enforced_w < MIN_AC_POWER_W:
        warns.append(
            f"Enforced GPU power limit {enforced_w:.0f} W < {MIN_AC_POWER_W:.0f} W: "
            "GPU is throttled (battery saver / OEM profile). Latency SLA at risk."
        )
    return warns


@dataclass
class HardwareReport:
    gpu_name: str
    capability: tuple[int, int]
    driver_version: str
    torch_version: str
    cuda_version: str
    total_vram_mb: float
    free_vram_mb: float
    ac_online: Optional[bool]
    battery_percent: Optional[int]
    power: dict
    bf16_ok: bool
    fp8_scaled_mm_ok: bool
    fp8_detail: str
    warnings: list[str] = field(default_factory=list)

    @property
    def is_target_sm120(self) -> bool:
        return self.capability == TARGET_CAPABILITY

    def summary(self) -> str:
        lines = [
            f"GPU           : {self.gpu_name} (sm_{self.capability[0]}{self.capability[1]})",
            f"Driver        : {self.driver_version} | torch {self.torch_version} | CUDA {self.cuda_version}",
            f"VRAM          : {self.total_vram_mb:.0f} MB total, {self.free_vram_mb:.0f} MB free",
            f"AC adapter    : {self.ac_online} (battery {self.battery_percent}%)",
            f"Power limits  : {self.power}",
            f"BF16 matmul   : {'OK' if self.bf16_ok else 'FAIL'}",
            f"FP8 scaled_mm : {'OK' if self.fp8_scaled_mm_ok else 'FAIL'} ({self.fp8_detail})",
        ]
        lines += [f"WARNING       : {w}" for w in self.warnings]
        return "\n".join(lines)


def _check_bf16(device: torch.device) -> bool:
    a = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
    b = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
    out = a @ b
    torch.cuda.synchronize()
    return bool(torch.isfinite(out.float()).all().item())


def _check_fp8(device: torch.device) -> tuple[bool, str]:
    x = torch.randn(128, 256, device=device)
    w = torch.randn(64, 256, device=device)
    x8, w8 = x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
    one = torch.ones(1, device=device)
    try:
        y = torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        if not torch.isfinite(y.float()).all().item():
            return False, "non-finite output"
        ref = x @ w.t()
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        return True, f"rel_err={rel:.4f}"
    except Exception as exc:  # noqa: BLE001 - report the real driver/kernel error
        return False, f"{type(exc).__name__}: {exc}"


def probe_hardware(require_sm120: bool = True) -> HardwareReport:
    if not torch.cuda.is_available():
        raise HardwareError("CUDA unavailable: AXION-3D requires the local NVIDIA GPU")
    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability(device)
    handle = _nvml_handle()
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    ac_online, battery_pct = _windows_ac_status()
    power = _power_limits_w(handle)
    bf16_ok = _check_bf16(device)
    fp8_ok, fp8_detail = _check_fp8(device)
    warns = power_warnings(ac_online, power["enforced_w"])
    if require_sm120 and cap != TARGET_CAPABILITY:
        raise HardwareError(
            f"Expected sm_120 (RTX 5060 Blackwell), found sm_{cap[0]}{cap[1]}. "
            "Dense NVFP4 kernels are validated for sm_120 only."
        )
    if not bf16_ok:
        raise HardwareError("BF16 matmul produced non-finite results on this GPU")
    return HardwareReport(
        gpu_name=torch.cuda.get_device_name(device),
        capability=cap,
        driver_version=pynvml.nvmlSystemGetDriverVersion(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "?",
        total_vram_mb=mem.total / _MB,
        free_vram_mb=mem.free / _MB,
        ac_online=ac_online,
        battery_percent=battery_pct,
        power=power,
        bf16_ok=bf16_ok,
        fp8_scaled_mm_ok=fp8_ok,
        fp8_detail=fp8_detail,
        warnings=warns,
    )


class VRAMBudgetGuard:
    """Hard cap on process VRAM: PyTorch peak AND NVML process memory AND
    Windows shared-RAM spill. Raises VRAMBudgetExceeded / SharedSpillDetected.

    Usage:
        with VRAMBudgetGuard(max_mb=2048) as guard:
            guard.assert_can_allocate(1500)   # pre-flight: blocks BEFORE spill
            ... run inference ...
        print(guard.peak_mb)
    """

    def __init__(
        self,
        max_mb: float = 2048.0,
        check_shared: bool = True,
        check_nvml: bool = True,
    ) -> None:
        if not torch.cuda.is_available():
            raise HardwareError("VRAMBudgetGuard requires CUDA")
        self.max_mb = float(max_mb)
        self.check_shared = check_shared
        self.check_nvml = check_nvml
        self.peak_mb: float = 0.0
        self.shared_spill_mb: float = 0.0
        self._baseline_shared_mb: Optional[float] = None
        self._active = False

    def _torch_peak_mb(self) -> float:
        return torch.cuda.max_memory_allocated() / _MB

    def assert_can_allocate(self, mb: float) -> None:
        """Pre-allocation hard stop so we never even touch shared RAM."""
        projected = torch.cuda.memory_allocated() / _MB + mb
        if projected > self.max_mb:
            raise VRAMBudgetExceeded(projected, self.max_mb, "pre-flight projection")
        free_mb, _total = torch.cuda.mem_get_info()
        if mb > free_mb / _MB:
            raise VRAMBudgetExceeded(mb, free_mb / _MB, "device free memory")

    def check(self) -> None:
        torch.cuda.synchronize()
        current = self._torch_peak_mb()
        if current > self.max_mb:
            raise VRAMBudgetExceeded(current, self.max_mb, "torch peak")

    def __enter__(self) -> "VRAMBudgetGuard":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if self.check_shared:
            self._baseline_shared_mb = shared_gpu_spill_mb()
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._active = False
        torch.cuda.synchronize()
        peak = self._torch_peak_mb()
        nvml_mb = nvml_process_vram_mb() if self.check_nvml else None
        if nvml_mb is not None:
            peak = max(peak, nvml_mb)
        self.peak_mb = peak
        if self.check_shared:
            end_shared = shared_gpu_spill_mb()
            baseline = self._baseline_shared_mb or 0.0
            self.shared_spill_mb = max(0.0, end_shared - baseline)
        if exc_type is not None:
            return False  # never mask the original error
        if peak > self.max_mb:
            raise VRAMBudgetExceeded(peak, self.max_mb, "torch peak + NVML process")
        if self.check_shared and self.shared_spill_mb > 0.0:
            raise SharedSpillDetected(self.shared_spill_mb)
        return False