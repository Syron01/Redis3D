"""NativeZero Multi-Modal Conditioner (roadfinal.md §3.3, §4.1, §7 Phase 5).

Clean-room, 100% local image + text conditioning encoder:
  - Vision branch: Lightweight ViT patch embedder + Transformer encoder for
    de-lighted images (RGB or 6-ch albedo+normal).
  - Text branch: UTF-8 byte-level tokenizer (vocab=256 + special tokens) +
    causal/bidirectional Transformer encoder.
  - Produces conditioning token sequence c [B, N_c, D_c] and pooled vector c_pool [B, D_c].
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (QAT-friendly, zero mean-shift)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class TransformerBlock(nn.Module):
    """Pre-norm dense Transformer block with SwiGLU-style MLP."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 2.5):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        h2 = self.norm2(x)
        a, b = self.fc1(h2).chunk(2, dim=-1)
        x = x + self.fc2(F.silu(a) * b)
        return x


class NativeZeroConditioner(nn.Module):
    """Local ViT + Byte-Transformer conditioner for image & text prompts."""

    PAD_TOKEN = 256
    BOS_TOKEN = 257
    VOCAB_SIZE = 258

    def __init__(
        self,
        cond_dim: int = 256,
        image_channels: int = 3,
        image_size: int = 128,
        patch_size: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
        max_text_len: int = 64,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_text_len = max_text_len

        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            image_channels, cond_dim, kernel_size=patch_size, stride=patch_size
        )
        self.img_pos_embed = nn.Parameter(torch.randn(1, num_patches, cond_dim) * 0.02)

        self.text_embed = nn.Embedding(self.VOCAB_SIZE, cond_dim, padding_idx=self.PAD_TOKEN)
        self.text_pos_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim) * 0.02)

        self.cls_token = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cond_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.out_norm = RMSNorm(cond_dim)

    @classmethod
    def tokenize_texts(
        cls,
        prompts: list[str],
        max_len: int = 64,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode UTF-8 strings into fixed-length byte token tensors."""
        tokens = torch.full((len(prompts), max_len), cls.PAD_TOKEN, dtype=torch.long)
        for i, text in enumerate(prompts):
            raw = text.encode("utf-8", errors="ignore")[: max_len - 1]
            tokens[i, 0] = cls.BOS_TOKEN
            for j, byte_val in enumerate(raw):
                tokens[i, j + 1] = byte_val
        if device is not None:
            tokens = tokens.to(device)
        return tokens

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        prompts: Optional[list[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode optional image and/or text into (cond_tokens, cond_pooled).

        Args:
            images: [B, C, H, W] normalized image tensor.
            text_tokens: [B, T] integer token ids.
            prompts: Optional list of prompt strings if text_tokens is None.

        Returns:
            cond_tokens: [B, N_c, cond_dim]
            cond_pooled: [B, cond_dim]
        """
        if images is None and text_tokens is None and prompts is None:
            raise ValueError("At least one of images, text_tokens, or prompts must be provided.")

        device = images.device if images is not None else (
            text_tokens.device if text_tokens is not None else next(self.parameters()).device
        )

        if text_tokens is None and prompts is not None:
            text_tokens = self.tokenize_texts(prompts, max_len=self.max_text_len, device=device)

        bsz = images.shape[0] if images is not None else text_tokens.shape[0]
        seq_parts = [self.cls_token.expand(bsz, -1, -1)]
        pad_parts = [torch.zeros((bsz, 1), dtype=torch.bool, device=device)]

        if images is not None:
            if images.shape[-2:] != (self.image_size, self.image_size):
                images = F.interpolate(
                    images,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            patches = self.patch_embed(images).flatten(2).transpose(1, 2)
            patches = patches + self.img_pos_embed[:, : patches.shape[1], :]
            seq_parts.append(patches)
            pad_parts.append(torch.zeros((bsz, patches.shape[1]), dtype=torch.bool, device=device))

        if text_tokens is not None:
            t_len = min(text_tokens.shape[1], self.max_text_len)
            t_ids = text_tokens[:, :t_len]
            t_emb = self.text_embed(t_ids) + self.text_pos_embed[:, :t_len, :]
            seq_parts.append(t_emb)
            pad_parts.append(t_ids == self.PAD_TOKEN)

        x = torch.cat(seq_parts, dim=1)
        pad_mask = torch.cat(pad_parts, dim=1)

        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask)

        x = self.out_norm(x)
        cond_pooled = x[:, 0, :]
        return x, cond_pooled
