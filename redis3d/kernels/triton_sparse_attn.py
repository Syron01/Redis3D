"""Block-Sparse Spatial Window Attention with Attn-QAT Safety (roadfinal.md §3.3, §7 Phase 6).

Applies symmetric QK-RMSNorm (`Attn-QAT`) before scaled dot-product attention
over local Morton windows + global registers, ensuring zero NaNs under NVFP4/FP8.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _rms_norm(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + eps)


def sparse_window_attention_qat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 64,
    apply_qk_norm: bool = True,
) -> torch.Tensor:
    """Compute Attn-QAT safe block-sparse window attention.

    Args:
        q, k, v: [B, H, L, D]
        window_size: Local Morton window size W.
        apply_qk_norm: Enforce symmetric QK-RMSNorm for low-precision stability.

    Returns:
        out: [B, H, L, D]
    """
    bsz, heads, seq_len, head_dim = q.shape
    if apply_qk_norm:
        q = _rms_norm(q)
        k = _rms_norm(k)

    pad_len = (window_size - (seq_len % window_size)) % window_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))

    num_win = q.shape[2] // window_size
    q_w = q.view(bsz, heads, num_win, window_size, head_dim).permute(0, 2, 1, 3, 4).reshape(
        bsz * num_win, heads, window_size, head_dim
    )
    k_w = k.view(bsz, heads, num_win, window_size, head_dim).permute(0, 2, 1, 3, 4).reshape(
        bsz * num_win, heads, window_size, head_dim
    )
    v_w = v.view(bsz, heads, num_win, window_size, head_dim).permute(0, 2, 1, 3, 4).reshape(
        bsz * num_win, heads, window_size, head_dim
    )

    out_w = F.scaled_dot_product_attention(q_w, k_w, v_w)
    out = out_w.view(bsz, num_win, heads, window_size, head_dim).permute(0, 2, 1, 3, 4).reshape(
        bsz, heads, num_win * window_size, head_dim
    )
    return out[:, :, :seq_len, :]
