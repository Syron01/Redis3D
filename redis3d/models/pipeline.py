"""Full Redis3DProPipeline (Resident & Streaming Modes) (roadfinal.md §3.4, §4.1, §7 Phase 7).

End-to-End Closed-Loop Pipeline:
  1. PIXL-Intrinsic De-Lighting (< 25 ms) -> (albedo, shading, normal, edges)
  2. NativeZeroConditioner -> (cond_tokens, cond_pooled)
  3. Stage 2 OT-MeanFlow-3D parallel K=4 Sobol rollout (1-NFE or 2-NFE)
  4. SW-VAE Decoder -> K=4 candidate dual-grid clouds
  5. Consilience-3D Best-of-4 Consensus & Physical Ranking
  6. PATR-2 Differentiable QEF & Intrinsic Texture Lock (6 Adam steps)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from redis3d.core.hardware_guard import VRAMBudgetGuard
from redis3d.models.conditioner import NativeZeroConditioner
from redis3d.models.consilience_verifier import (
    Consilience3DVerifier,
    generate_sobol_latents,
)
from redis3d.models.intrinsic_delight import create_pixl_intrinsic
from redis3d.models.latent_meanflow import Stage2LatentMeanFlow, apply_sym_lock
from redis3d.models.sw_vae import LATENT_DIM, SparseWaveletVAE
from redis3d.refinement.patr2 import PATR2Refiner


@dataclass
class PipelineOutput:
    coords: torch.Tensor       # [L, 3]
    normals: torch.Tensor      # [L, 2] octahedral
    pbr: torch.Tensor          # [L, 7]
    ranks: torch.Tensor        # [L]
    best_candidate_idx: int
    consilience_scores: torch.Tensor
    initial_iou: float
    final_iou: float


class Redis3DProPipeline:
    """Closed-loop 99%-accuracy 3D generation pipeline (< 2 GB VRAM)."""

    def __init__(
        self,
        delighter: Optional[torch.nn.Module] = None,
        conditioner: Optional[NativeZeroConditioner] = None,
        stage2: Optional[Stage2LatentMeanFlow] = None,
        vae: Optional[SparseWaveletVAE] = None,
        vram_mode: str = "resident",
        max_vram_mb: int = 2048,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.vram_mode = vram_mode
        self.max_vram_mb = max_vram_mb

        self.delighter = delighter or create_pixl_intrinsic(base_channels=48)
        self.conditioner = conditioner or NativeZeroConditioner(
            cond_dim=256, image_size=128, patch_size=16, num_layers=4, num_heads=8
        )
        self.stage2 = stage2 or Stage2LatentMeanFlow(
            latent_dim=LATENT_DIM,
            model_dim=256,
            cond_dim=256,
            num_groups=2,
            num_heads=8,
            window_size=64,
            num_registers=16,
        )
        self.vae = vae or SparseWaveletVAE(embed_dim=128, latent_dim=LATENT_DIM, num_layers=3)

        self.verifier = Consilience3DVerifier()
        self.patr2 = PATR2Refiner(num_steps=6)

        if vram_mode == "resident":
            self.delighter.to(self.device).eval()
            self.conditioner.to(self.device).eval()
            self.stage2.to(self.device).eval()
            self.vae.to(self.device).eval()
        else:
            # Streaming mode: keep modules on pinned/CPU until needed
            self.delighter.cpu().eval()
            self.conditioner.cpu().eval()
            self.stage2.cpu().eval()
            self.vae.cpu().eval()

    def _run_module(self, mod: torch.nn.Module, fn):
        if self.vram_mode == "streaming":
            mod.to(self.device)
            try:
                return fn()
            finally:
                mod.cpu()
                torch.cuda.empty_cache()
        return fn()

    def generate(
        self,
        image: torch.Tensor,
        prompt: str = "",
        ref_mask: Optional[torch.Tensor] = None,
        num_candidates: int = 4,
        nfe_steps: int = 1,
        patr_steps: int = 6,
        seq_len: int = 256,
        sym_lock: bool = False,
        seed: int = 42,
    ) -> PipelineOutput:
        """Run full closed-loop generation inside VRAMBudgetGuard."""
        self.patr2.num_steps = patr_steps

        with VRAMBudgetGuard(max_mb=self.max_vram_mb):
            img_dev = image.to(self.device)
            if img_dev.dim() == 3:
                img_dev = img_dev.unsqueeze(0)

            # 1. PIXL-Intrinsic De-Lighter
            with torch.no_grad():
                intrinsic = self._run_module(self.delighter, lambda: self.delighter(img_dev))
            albedo = intrinsic.albedo
            edges = intrinsic.edges

            if ref_mask is None:
                # Derive foreground mask from non-background brightness
                ref_mask_t = (img_dev[0].mean(dim=0) > 0.05).float()
            else:
                ref_mask_t = ref_mask.to(self.device)

            # 2. NativeZeroConditioner
            with torch.no_grad():
                c_tokens, c_pooled = self._run_module(
                    self.conditioner,
                    lambda: self.conditioner(images=albedo, prompts=[prompt]),
                )

            # 3. Parallel K=4 Sobol Rollout on Stage 2 OT-MeanFlow-3D
            z0 = generate_sobol_latents(
                num_candidates=num_candidates,
                seq_len=seq_len,
                latent_dim=LATENT_DIM,
                device=self.device,
                seed=seed,
            )
            c_pool_k = c_pooled.expand(num_candidates, -1)
            c_tok_k = c_tokens.expand(num_candidates, -1, -1)

            with torch.no_grad():
                if nfe_steps <= 1:
                    z1_k = self._run_module(
                        self.stage2,
                        lambda: self.stage2.sample_1nfe(
                            z0, c_pool_k, cond_tokens=c_tok_k, sym_lock=sym_lock
                        ),
                    )
                else:
                    z1_k = self._run_module(
                        self.stage2,
                        lambda: self.stage2.sample_2nfe(
                            z0, c_pool_k, cond_tokens=c_tok_k, sym_lock=sym_lock
                        ),
                    )

                # 4. SW-VAE Decode all K candidates
                recon = self._run_module(self.vae, lambda: self.vae.decode(z1_k))

            # 5. Consilience-3D Best-of-4 Consensus
            cons_res = self.verifier.score_and_select(
                coords=recon.coords,
                ref_mask=ref_mask_t,
                ref_edges=edges[0, 0] if edges.dim() == 4 else edges,
                rank_logits=recon.rank_logits,
            )
            k_star = cons_res.best_idx

            best_coords = recon.coords[k_star]
            best_normals = recon.normals[k_star]
            best_pbr = recon.pbr[k_star]
            best_ranks = torch.argmax(recon.rank_logits[k_star], dim=-1)

            # 6. PATR-2 Pixel-Lock Refinement
            if patr_steps > 0:
                patr_res = self.patr2.refine(
                    coords=best_coords,
                    pbr=best_pbr,
                    ref_mask=ref_mask_t,
                    ref_albedo=albedo[0],
                    ref_edges=edges[0, 0] if edges.dim() == 4 else edges,
                    ranks=best_ranks,
                )
                final_coords = patr_res.coords
                final_pbr = patr_res.pbr
                init_iou = patr_res.initial_iou
                final_iou = patr_res.final_iou
            else:
                final_coords = best_coords
                final_pbr = best_pbr
                init_iou = float(cons_res.silhouette_iou[k_star].item())
                final_iou = init_iou

            if sym_lock:
                final_coords = apply_sym_lock(final_coords, sym_axis=0)

            return PipelineOutput(
                coords=final_coords,
                normals=best_normals,
                pbr=final_pbr,
                ranks=best_ranks,
                best_candidate_idx=k_star,
                consilience_scores=cons_res.scores,
                initial_iou=init_iou,
                final_iou=final_iou,
            )
