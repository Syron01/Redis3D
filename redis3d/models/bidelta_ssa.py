"""BiDelta-SSA Dense Backbone (roadfinal.md §3.3, §7 Phase 5).

Strictly Dense 3:1 Interleaved Architecture (ZERO MoE Routing — sm_120 safe!):
  - 75% (3 of every 4 blocks): Bidirectional Gated-DeltaNet (`BiDeltaBlock`)
    running chunked O(L) linear state-space scans over forward and reverse
    Morton-ordered token sequences.
  - 25% (1 of every 4 blocks): Spatial Sparse Window Attention (`SSABlock`)
    over local 3D Morton windows (W=64) + 16 global register tokens, with
    symmetric QK-RMSNorm (`Attn-QAT` rule) for NVFP4 quantization stability.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from redis3d.models.conditioner import RMSNorm


class AdaLNZero(nn.Module):
    """Adaptive Layer/RMSNorm Zero modulation from conditioning vector."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(cond_dim, dim * 3)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.proj(F.silu(cond)).unsqueeze(1).chunk(3, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift
        return h, gate


def _chunked_gated_delta_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """O(L)-memory chunked Gated-DeltaNet linear state-space scan.

    Args:
        q, k: [B, H, L, D_k]
        v:    [B, H, L, D_v]
        alpha: [B, H, L, 1] decay gate in (0, 1)
        beta:  [B, H, L, 1] write/delta gate in (0, 1)
        chunk_size: Local parallel chunk length C (memory is O(L * C), linear in L).

    Returns:
        out: [B, H, L, D_v]
    """
    bsz, heads, seq_len, dk = q.shape
    dv = v.shape[-1]

    pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        alpha = F.pad(alpha, (0, 0, 0, pad_len), value=1.0)
        beta = F.pad(beta, (0, 0, 0, pad_len), value=0.0)

    total_len = q.shape[2]
    num_chunks = total_len // chunk_size

    q_c = q.view(bsz, heads, num_chunks, chunk_size, dk)
    k_c = k.view(bsz, heads, num_chunks, chunk_size, dk)
    v_c = v.view(bsz, heads, num_chunks, chunk_size, dv)
    log_a = torch.log(alpha.clamp(min=1e-4, max=1.0)).view(bsz, heads, num_chunks, chunk_size)
    b_c = beta.view(bsz, heads, num_chunks, chunk_size, 1)

    cum_log_a = torch.cumsum(log_a, dim=-1)  # [B, H, N, C]
    causal_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool)
    )

    state = torch.zeros(bsz, heads, dk, dv, device=q.device, dtype=q.dtype)
    out_chunks = []

    for idx in range(num_chunks):
        qi = q_c[:, :, idx]          # [B, H, C, D_k]
        ki = k_c[:, :, idx]          # [B, H, C, D_k]
        vi = v_c[:, :, idx]          # [B, H, C, D_v]
        bi = b_c[:, :, idx]          # [B, H, C, 1]
        cla = cum_log_a[:, :, idx]   # [B, H, C]

        # Delta rule correction against incoming chunk state
        pred_v = torch.matmul(ki, state)  # [B, H, C, D_v]
        v_eff = bi * (vi - pred_v)        # [B, H, C, D_v]

        # Inter-chunk contribution from prior state: exp(cla) * (qi @ state)
        decay_from_start = torch.exp(cla).unsqueeze(-1)  # [B, H, C, 1]
        inter_out = decay_from_start * torch.matmul(qi, state)

        # Intra-chunk causal attention with exponential decay
        # diff[m, n] = cla[m] - cla[n] for m >= n
        diff = (cla.unsqueeze(-1) - cla.unsqueeze(-2)).clamp(min=-20.0, max=0.0)
        decay_mat = torch.where(causal_mask, torch.exp(diff), torch.zeros_like(diff))
        qk = torch.matmul(qi, ki.transpose(-1, -2)) * decay_mat
        intra_out = torch.matmul(qk, v_eff)

        out_chunks.append(inter_out + intra_out)

        # Update recurrent state for next chunk
        cla_end = cla[:, :, -1:]  # [B, H, 1]
        decay_to_end = torch.exp((cla_end - cla).clamp(min=-20.0, max=0.0)).unsqueeze(-1)
        state_decay = torch.exp(cla_end).unsqueeze(-1)  # [B, H, 1, 1]
        state = state_decay * state + torch.matmul(
            (ki * decay_to_end).transpose(-1, -2), v_eff
        )

    out = torch.cat(out_chunks, dim=2)
    return out[:, :, :seq_len, :]


class BiDeltaBlock(nn.Module):
    """Bidirectional Gated-DeltaNet block with O(L) memory complexity."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int = 8,
        chunk_size: int = 64,
        mlp_ratio: float = 2.5,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size

        self.adaln1 = AdaLNZero(dim, cond_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.gates = nn.Linear(dim, num_heads * 2)  # alpha, beta per head
        self.out_proj = nn.Linear(dim, dim)

        self.adaln2 = AdaLNZero(dim, cond_dim)
        hidden = int(dim * mlp_ratio)
        self.mlp_in = nn.Linear(dim, hidden * 2)
        self.mlp_out = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        h, gate1 = self.adaln1(x, cond)

        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q_raw = F.silu(q).view(bsz, seq_len, self.num_heads, self.head_dim)
        k_raw = F.silu(k).view(bsz, seq_len, self.num_heads, self.head_dim)
        q = (q_raw * torch.rsqrt(q_raw.pow(2).sum(dim=-1, keepdim=True) + 1e-4)).transpose(1, 2)
        k = (k_raw * torch.rsqrt(k_raw.pow(2).sum(dim=-1, keepdim=True) + 1e-4)).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        g_ab = self.gates(h).view(bsz, seq_len, self.num_heads, 2).permute(0, 2, 1, 3)
        alpha = torch.sigmoid(g_ab[..., 0:1] + 2.0)  # biased toward high retention
        beta = torch.sigmoid(g_ab[..., 1:2]) / math.sqrt(float(self.chunk_size))

        # Forward Morton scan + Reverse Morton scan (Bidirectional)
        y_fwd = _chunked_gated_delta_scan(q, k, v, alpha, beta, chunk_size=self.chunk_size)
        y_rev = _chunked_gated_delta_scan(
            torch.flip(q, dims=[2]),
            torch.flip(k, dims=[2]),
            torch.flip(v, dims=[2]),
            torch.flip(alpha, dims=[2]),
            torch.flip(beta, dims=[2]),
            chunk_size=self.chunk_size,
        )
        y_rev = torch.flip(y_rev, dims=[2])

        y = 0.5 * (y_fwd + y_rev)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        x = x + gate1 * self.out_proj(y)

        h2, gate2 = self.adaln2(x, cond)
        a, b = self.mlp_in(h2).chunk(2, dim=-1)
        x = x + gate2 * self.mlp_out(F.silu(a) * b)
        return x


class SSABlock(nn.Module):
    """Spatial Sparse Window Attention (W=64) + 16 Global Register Tokens.

    Follows the Attn-QAT rule: symmetric QK-RMSNorm keeps attention logits
    well-conditioned for NVFP4/FP8 quantization.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int = 8,
        window_size: int = 64,
        num_registers: int = 16,
        mlp_ratio: float = 2.5,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.num_registers = num_registers

        self.adaln1 = AdaLNZero(dim, cond_dim)
        self.registers = nn.Parameter(torch.randn(1, num_registers, dim) * 0.02)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(dim, dim)

        self.adaln2 = AdaLNZero(dim, cond_dim)
        hidden = int(dim * mlp_ratio)
        self.mlp_in = nn.Linear(dim, hidden * 2)
        self.mlp_out = nn.Linear(hidden, dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cond_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        h, gate1 = self.adaln1(x, cond)

        # Update & broadcast global registers + optional conditioning tokens
        regs = self.registers.expand(bsz, -1, -1)
        if cond_tokens is not None:
            # Summarize conditioning tokens into register space via cross-attn
            reg_q = self.q_norm(
                self.q_proj(regs).view(bsz, self.num_registers, self.num_heads, self.head_dim).transpose(1, 2)
            )
            ck = self.k_norm(
                self.k_proj(cond_tokens).view(bsz, cond_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
            )
            cv = self.v_proj(cond_tokens).view(bsz, cond_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
            regs_ctx = F.scaled_dot_product_attention(reg_q, ck, cv)
            regs = regs + regs_ctx.transpose(1, 2).contiguous().view(bsz, self.num_registers, self.dim)

        # Pad sequence to multiple of window_size
        w = self.window_size
        pad_len = (w - (seq_len % w)) % w
        h_pad = F.pad(h, (0, 0, 0, pad_len)) if pad_len > 0 else h
        num_win = h_pad.shape[1] // w

        # Reshape to [B * num_win, W, D]
        h_win = h_pad.view(bsz, num_win, w, self.dim).reshape(bsz * num_win, w, self.dim)
        regs_win = regs.unsqueeze(1).expand(bsz, num_win, self.num_registers, self.dim).reshape(
            bsz * num_win, self.num_registers, self.dim
        )

        kv_tokens = torch.cat([regs_win, h_win], dim=1)  # [B*num_win, R + W, D]

        q = self.q_proj(h_win).view(bsz * num_win, w, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_tokens).view(bsz * num_win, self.num_registers + w, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_tokens).view(bsz * num_win, self.num_registers + w, self.num_heads, self.head_dim).transpose(1, 2)

        # Attn-QAT symmetric QK-RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, num_win * w, self.dim)[:, :seq_len, :]
        x = x + gate1 * self.out_proj(attn_out)

        h2, gate2 = self.adaln2(x, cond)
        a, b = self.mlp_in(h2).chunk(2, dim=-1)
        x = x + gate2 * self.mlp_out(F.silu(a) * b)
        return x


class BiDeltaSSABackbone(nn.Module):
    """Dense 3:1 Interleaved BiDelta-SSA Backbone (Zero MoE)."""

    def __init__(
        self,
        dim: int = 256,
        cond_dim: int = 256,
        num_groups: int = 4,
        num_heads: int = 8,
        window_size: int = 64,
        num_registers: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_groups = num_groups

        self.blocks = nn.ModuleList()
        for _ in range(num_groups):
            # 3:1 Interleaving: 3 BiDeltaBlocks + 1 SSABlock
            self.blocks.append(BiDeltaBlock(dim, cond_dim, num_heads=num_heads, chunk_size=window_size))
            self.blocks.append(BiDeltaBlock(dim, cond_dim, num_heads=num_heads, chunk_size=window_size))
            self.blocks.append(BiDeltaBlock(dim, cond_dim, num_heads=num_heads, chunk_size=window_size))
            self.blocks.append(
                SSABlock(dim, cond_dim, num_heads=num_heads, window_size=window_size, num_registers=num_registers)
            )

        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cond_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for blk in self.blocks:
            if isinstance(blk, SSABlock):
                x = blk(x, cond, cond_tokens=cond_tokens)
            else:
                x = blk(x, cond)
        return self.final_norm(x)
