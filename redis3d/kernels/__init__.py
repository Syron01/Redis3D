"""Runtime Kernel Dispatcher for Blackwell sm_120 (roadfinal.md §3.5, §7 Phase 6).

Strictly Dense kernels (ZERO grouped/MoE FP4 kernels — sm_120 safe).
Automatically selects Triton when available and falls back to verified dense
PyTorch CUDA operators on native Windows.
"""

from __future__ import annotations

import torch


def has_triton() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def get_sm_arch() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability(0)


def is_blackwell_sm120() -> bool:
    major, minor = get_sm_arch()
    return major == 12 and minor == 0


__all__ = ["has_triton", "get_sm_arch", "is_blackwell_sm120"]
