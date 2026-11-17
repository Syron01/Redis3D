"""AWQ-Weighted NVFP4 + BiLoCo Calibration & Sensitive-Layer Shielding (roadfinal.md §3.5, §7 Phase 6).

Shields sensitive layers (coordinate/QEF/normal heads, embeddings, time/CFG MLPs,
and norms) in FP16/BF16/FP32 while compressing backbone Linear layers to
NVFP4 E2M1 + SVDQuant r16 + Binary ±1 r256 BiLoCo.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from redis3d.kernels.nvfp4_biloco import NVFP4BiLoCoLinear

DEFAULT_SHIELDED_PATTERNS = (
    "head_coords",
    "head_normals",
    "head_pbr",
    "head_rank",
    "in_proj",
    "out_proj",
    "time_mlp",
    "time_cfg_mlp",
    "patch_embed",
    "text_embed",
    "fc_mean",
    "fc_logvar",
    "embedder",
)


def is_shielded_layer(name: str, shielded_patterns: Iterable[str] = DEFAULT_SHIELDED_PATTERNS) -> bool:
    """Return True if module name matches a sensitive layer that must remain FP16/FP32."""
    lname = name.lower()
    return any(pat.lower() in lname for pat in shielded_patterns)


@torch.no_grad()
def collect_activation_scales(
    model: nn.Module,
    sample_fn,
    shielded_patterns: Iterable[str] = DEFAULT_SHIELDED_PATTERNS,
) -> dict[str, torch.Tensor]:
    """Collect per-channel mean absolute input activation scales for AWQ weighting."""
    scales: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    def _make_hook(layer_name: str):
        def _hook(_mod, inp, _out):
            if not inp:
                return
            x = inp[0].detach().float()
            x_flat = x.reshape(-1, x.shape[-1])
            ch_mean = x_flat.abs().mean(dim=0)
            if layer_name not in scales:
                scales[layer_name] = ch_mean
                counts[layer_name] = 1
            else:
                scales[layer_name] += ch_mean
                counts[layer_name] += 1

        return _hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not is_shielded_layer(name, shielded_patterns):
            hooks.append(mod.register_forward_hook(_make_hook(name)))

    try:
        model.eval()
        sample_fn(model)
    finally:
        for h in hooks:
            h.remove()

    for name in scales:
        scales[name] = scales[name] / max(1, counts[name])
    return scales


def quantize_model_nvfp4_biloco(
    model: nn.Module,
    act_scales: Optional[dict[str, torch.Tensor]] = None,
    svd_rank: int = 16,
    biloco_rank: int = 256,
    min_features: int = 64,
    shielded_patterns: Iterable[str] = DEFAULT_SHIELDED_PATTERNS,
) -> tuple[nn.Module, dict[str, int]]:
    """Replace eligible backbone nn.Linear layers with NVFP4BiLoCoLinear in-place."""
    stats = {"quantized": 0, "shielded": 0}

    def _replace(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                if (
                    is_shielded_layer(full_name, shielded_patterns)
                    or child.in_features < min_features
                    or child.out_features < min_features
                ):
                    stats["shielded"] += 1
                else:
                    s = act_scales.get(full_name) if act_scales is not None else None
                    q_layer = NVFP4BiLoCoLinear.from_linear(
                        child,
                        svd_rank=svd_rank,
                        biloco_rank=biloco_rank,
                        act_scales=s,
                    )
                    setattr(parent, child_name, q_layer)
                    stats["quantized"] += 1
            else:
                _replace(child, full_name)

    _replace(model)
    return model, stats
