"""Real latency + VRAM telemetry (perf_counter, NVML, Windows GPU perf counters).

No synthetic numbers anywhere: every value is read from the OS / driver / PyTorch.
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

import pynvml
import torch

_MB = 1024 * 1024
_nvml_ready = False


def _nvml_handle() -> "pynvml.LP_nvmlDevice_t":
    global _nvml_ready
    if not _nvml_ready:
        pynvml.nvmlInit()
        _nvml_ready = True
    idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
    return pynvml.nvmlDeviceGetHandleByIndex(idx)


def shared_gpu_spill_mb(pid: Optional[int] = None) -> float:
    """Windows 'GPU Process Memory\\Shared Usage' for one process, in MB.

    This is the Task-Manager 'Shared GPU memory' figure: system RAM borrowed by
    the GPU. A healthy CUDA inference process must read 0 here.
    """
    pid = os.getpid() if pid is None else pid
    out = subprocess.run(
        ["typeperf", r"\GPU Process Memory(*)\Shared Usage", "-sc", "1"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if out.returncode != 0:
        raise RuntimeError(f"typeperf failed: {out.stderr.strip()[:200]}")
    rows = list(csv.reader(io.StringIO(out.stdout)))
    if len(rows) < 2:
        raise RuntimeError("typeperf returned no sample rows")
    header, values = rows[0], rows[1]
    total_bytes = 0.0
    tag = f"pid_{pid}_"
    for name, val in zip(header[1:], values[1:]):
        if tag in name:
            try:
                total_bytes += float(val)
            except ValueError:
                total_bytes += float(val.replace(",", "."))
    return total_bytes / _MB


def nvml_process_vram_mb(pid: Optional[int] = None) -> Optional[float]:
    """Per-process dedicated VRAM from the driver (None if WDDM hides it)."""
    pid = os.getpid() if pid is None else pid
    handle = _nvml_handle()
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except pynvml.NVMLError:
        return None
    for p in procs:
        if p.pid == pid and p.usedGpuMemory is not None:
            return p.usedGpuMemory / _MB
    return None


@dataclass
class VRAMSnapshot:
    torch_allocated_mb: float
    torch_reserved_mb: float
    torch_peak_mb: float
    nvml_used_mb: float
    nvml_free_mb: float
    nvml_total_mb: float
    nvml_process_mb: Optional[float]
    shared_spill_mb: Optional[float]

    def as_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def vram_snapshot(include_shared: bool = True) -> VRAMSnapshot:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; telemetry requires the real GPU")
    torch.cuda.synchronize()
    handle = _nvml_handle()
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return VRAMSnapshot(
        torch_allocated_mb=torch.cuda.memory_allocated() / _MB,
        torch_reserved_mb=torch.cuda.memory_reserved() / _MB,
        torch_peak_mb=torch.cuda.max_memory_allocated() / _MB,
        nvml_used_mb=mem.used / _MB,
        nvml_free_mb=mem.free / _MB,
        nvml_total_mb=mem.total / _MB,
        nvml_process_mb=nvml_process_vram_mb(),
        shared_spill_mb=shared_gpu_spill_mb() if include_shared else None,
    )


class LatencyBreakdown:
    """Wall-clock stage timer. GPU stages synchronize so timings are real."""

    def __init__(self) -> None:
        self.stages_ms: dict[str, float] = {}
        self._t0 = time.perf_counter()

    def stage(self, name: str, sync_cuda: bool = True):
        return _Stage(self, name, sync_cuda)

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def report(self) -> dict:
        return {
            "stages_ms": {k: round(v, 3) for k, v in self.stages_ms.items()},
            "total_ms": round(self.total_ms, 3),
        }


class _Stage:
    def __init__(self, parent: LatencyBreakdown, name: str, sync_cuda: bool) -> None:
        self.parent, self.name, self.sync_cuda = parent, name, sync_cuda

    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.parent.stages_ms[self.name] = (time.perf_counter() - self._t0) * 1000.0
        return False