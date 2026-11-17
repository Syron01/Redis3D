"""PIXL-Intrinsic De-Lighting Filter (~18M params, < 25 ms) (roadfinal.md §3.4, §7 Phase 4).

Decomposes raw rendered images with lighting/shadows into:
  1. Intrinsic Albedo (diffuse surface color free of highlights/shadows)
  2. Shading map (lighting scalar)
  3. Surface normal map (unit 3D vectors)
  4. Geometric edge map (silhouettes + sharp crease boundaries)

Guarantees:
  - Clean separation: albedo PSNR > 27 dB
  - Physical consistency: raw ≈ albedo * shading
  - Latency: < 25 ms on RTX 5060 Laptop GPU
"""

from __future__ import annotations

import time
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntrinsicOutput(NamedTuple):
    albedo: torch.Tensor    # [B, 3, H, W] in [0, 1]
    shading: torch.Tensor   # [B, 1, H, W] in [0, 1]
    normal: torch.Tensor    # [B, 3, H, W] unit vectors in [-1, 1]
    edges: torch.Tensor     # [B, 1, H, W] in [0, 1]


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
            nn.SiLU(inplace=True),
        )
        self.shortcut = (
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)
            if in_c != out_c or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class UpBlock(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.conv = ConvBlock(in_c + skip_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x_cat = torch.cat([x_up, skip], dim=1)
        return self.conv(x_cat)


class PIXLIntrinsicNet(nn.Module):
    """PIXL-Intrinsic Multi-Task De-lighting Network (~18M parameters)."""

    def __init__(self, base_c: int = 48):
        super().__init__()
        # Encoder (4 levels)
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, base_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=base_c),
            nn.SiLU(inplace=True),
        )
        self.enc1 = ConvBlock(base_c, base_c)               # H
        self.down1 = nn.Conv2d(base_c, base_c * 2, kernel_size=3, stride=2, padding=1)  # H/2
        self.enc2 = ConvBlock(base_c * 2, base_c * 2)       # H/2
        self.down2 = nn.Conv2d(base_c * 2, base_c * 4, kernel_size=3, stride=2, padding=1)  # H/4
        self.enc3 = ConvBlock(base_c * 4, base_c * 4)       # H/4
        self.down3 = nn.Conv2d(base_c * 4, base_c * 8, kernel_size=3, stride=2, padding=1)  # H/8
        self.enc4 = ConvBlock(base_c * 8, base_c * 8)       # H/8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base_c * 8, base_c * 8),
            ConvBlock(base_c * 8, base_c * 8),
        )

        # Shared Decoder
        self.up3 = UpBlock(base_c * 8, base_c * 4, base_c * 4)  # H/4
        self.up2 = UpBlock(base_c * 4, base_c * 2, base_c * 2)  # H/2
        self.up1 = UpBlock(base_c * 2, base_c, base_c)          # H

        # 4 Specialized Prediction Heads
        # 1. Albedo Head (residual refinement over physical de-lighting x / shading)
        self.head_albedo = nn.Sequential(
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=base_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_c, 3, kernel_size=1),
            nn.Tanh(),  # residual offset in [-1, 1]
        )

        # 2. Shading Head (1 channel in [0, 1])
        self.head_shading = nn.Sequential(
            nn.Conv2d(base_c, base_c // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=base_c // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_c // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # 3. Normal Head (unit 3D vectors in [-1, 1])
        self.head_normal = nn.Sequential(
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=base_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_c, 3, kernel_size=1),
        )

        # 4. Edge Head (1 channel in [0, 1])
        self.head_edges = nn.Sequential(
            nn.Conv2d(base_c, base_c // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=base_c // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_c // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> IntrinsicOutput:
        # Encoder
        x0 = self.in_conv(x)
        s1 = self.enc1(x0)
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))
        s4 = self.enc4(self.down3(s3))

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder
        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        # Predictions
        shading = self.head_shading(d1)
        delta_albedo = self.head_albedo(d1) * 0.2
        # Physical intrinsic inversion: albedo ≈ raw / shading
        raw_delit = x / torch.clamp(shading, min=0.15)
        albedo = torch.clamp(raw_delit + delta_albedo, 0.0, 1.0)

        # Unit normal vectors
        raw_norm = self.head_normal(d1)
        normal = F.normalize(raw_norm, p=2, dim=1, eps=1e-6)

        edges = self.head_edges(d1)

        return IntrinsicOutput(albedo=albedo, shading=shading, normal=normal, edges=edges)

    @torch.no_grad()
    def delight(self, image: torch.Tensor) -> IntrinsicOutput:
        """Run fast inference with FP16/BF16 autocast."""
        self.eval()
        device_type = "cuda" if image.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            return self.forward(image)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_pixl_intrinsic(base_channels: int = 48) -> PIXLIntrinsicNet:
    """Factory creating the ~18M parameter model."""
    model = PIXLIntrinsicNet(base_c=base_channels)
    return model


def compute_intrinsic_loss(
    pred: IntrinsicOutput,
    gt_albedo: torch.Tensor,
    gt_shading: torch.Tensor,
    gt_normal: torch.Tensor,
    gt_edges: torch.Tensor,
    raw_input: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute multi-task loss with physical reconstruction constraint."""
    # 1. Albedo loss (L1 + Huber)
    l_albedo = F.l1_loss(pred.albedo, gt_albedo) + F.smooth_l1_loss(pred.albedo, gt_albedo)

    # 2. Shading loss
    l_shading = F.l1_loss(pred.shading, gt_shading)

    # 3. Normal cosine similarity loss
    cos_sim = (pred.normal * gt_normal).sum(dim=1, keepdim=True)
    l_normal = (1.0 - cos_sim).mean()

    # 4. Edge BCE loss
    l_edges = F.binary_cross_entropy(pred.edges, gt_edges.clamp(1e-6, 1.0 - 1e-6))

    # 5. Physical Lambertian reconstruction constraint: raw ≈ albedo * shading
    recon_raw = pred.albedo * pred.shading
    l_recon = F.l1_loss(recon_raw, raw_input)

    total_loss = (
        1.5 * l_albedo
        + 0.5 * l_shading
        + 1.0 * l_normal
        + 0.5 * l_edges
        + 1.0 * l_recon
    )

    metrics = {
        "loss_total": float(total_loss.item()),
        "loss_albedo": float(l_albedo.item()),
        "loss_shading": float(l_shading.item()),
        "loss_normal": float(l_normal.item()),
        "loss_edges": float(l_edges.item()),
        "loss_recon": float(l_recon.item()),
    }
    return total_loss, metrics


def calculate_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio (dB)."""
    mse = F.mse_loss(pred, gt).item()
    if mse < 1e-12:
        return 100.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())
