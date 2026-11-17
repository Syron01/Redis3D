"""Occupancy-normalized 3D Haar-lifting wavelet compression (roadfinal.md §3.1.2).

Token layout (12 channels): [v_local(3), oct_normal(2), albedo(3), rough, metal,
opacity, emissive]. Geometry = channels 0:5, PBR = 5:12. Token key =
cell_morton_key * 2 + sheet_id (dual-sheet, K <= 2).

Prune children into a parent super-voxel ONLY IF:
  max|d_geom| < tau_geom  AND  max|d_pbr| < tau_pbr  AND  rank(A_parent) == 1.
Rank-2 creases and rank-3 corners are NEVER pruned.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from redis3d.math.morton import MortonIndex, morton_decode, morton_encode, token_key, token_split
from redis3d.math.qef import sym_eig3_desc

GEOM_DIM = 5
FEAT_DIM = 12


def rank_from_cov(cov: torch.Tensor, eps_plane: float = 0.01, eps_edge: float = 0.1) -> torch.Tensor:
    eig = sym_eig3_desc(cov)
    s1 = eig[:, 0].clamp_min(1e-12)
    r21, r31 = eig[:, 1] / s1, eig[:, 2] / s1
    rank = torch.full(eig.shape[:1], 3, dtype=torch.int64, device=eig.device)
    rank = torch.where(r31 < eps_edge, torch.full_like(rank, 2), rank)
    rank = torch.where(r21 < eps_plane, torch.full_like(rank, 1), rank)
    rank = torch.where(eig[:, 0] < 1e-9, torch.zeros_like(rank), rank)
    return rank


@dataclass
class PrunedOctree:
    keys: torch.Tensor          # surviving token keys (mixed levels)
    feats: torch.Tensor         # [S, 12] in each token's own cell frame
    covs: torch.Tensor          # [S, 3, 3]
    original_keys: torch.Tensor # [N] fine-level token keys, input order
    original_level: int
    ratio: float                # N / S token compression


def haar_lift_prune(
    keys: torch.Tensor,
    feats: torch.Tensor,
    covs: torch.Tensor,
    level: int,
    tau_geom: float = 0.02,
    tau_pbr: float = 0.02,
    eps_plane: float = 0.01,
    eps_edge: float = 0.1,
) -> PrunedOctree:
    device = keys.device
    original_keys = keys.clone()
    n_original = keys.numel()

    pool_keys, pool_feats, pool_covs = keys, feats, covs
    survivors_k, survivors_f, survivors_c = [], [], []

    for lev in range(level, 0, -1):
        if pool_keys.numel() == 0:
            break
        cell_part, sheet = token_split(pool_keys)
        coords, _ = morton_decode(cell_part)
        parent_cell = morton_encode(coords >> 1, lev - 1)
        group_key = token_key(parent_cell, sheet)
        uniq, inv = torch.unique(group_key, return_inverse=True)
        G = uniq.numel()

        # child vertex -> parent cell frame: v_par = (offset + v_child) / 2
        offset = (coords & 1).to(feats.dtype)
        f_par = pool_feats.clone()
        f_par[:, :3] = (offset + pool_feats[:, :3]) * 0.5

        cnt = torch.bincount(inv, minlength=G).to(feats.dtype).clamp_min(1.0)
        s = torch.zeros(G, FEAT_DIM, dtype=feats.dtype, device=device)
        s.index_add_(0, inv, f_par)
        s = s / cnt.unsqueeze(-1)

        d = f_par - s[inv]
        d_geom = d[:, :GEOM_DIM].abs().amax(dim=1)
        d_pbr = d[:, GEOM_DIM:].abs().amax(dim=1)
        max_g = torch.full((G,), -1.0, dtype=feats.dtype, device=device)
        max_p = torch.full((G,), -1.0, dtype=feats.dtype, device=device)
        max_g.scatter_reduce_(0, inv, d_geom, reduce="amax", include_self=False)
        max_p.scatter_reduce_(0, inv, d_pbr, reduce="amax", include_self=False)

        cov_sum = torch.zeros(G, 3, 3, dtype=covs.dtype, device=device)
        cov_sum.index_add_(0, inv, pool_covs)
        rank = rank_from_cov(cov_sum, eps_plane, eps_edge)

        prune = (max_g < tau_geom) & (max_p < tau_pbr) & (rank == 1)
        prune_tok = prune[inv]

        survivors_k.append(pool_keys[~prune_tok])
        survivors_f.append(pool_feats[~prune_tok])
        survivors_c.append(pool_covs[~prune_tok])

        pool_keys = uniq[prune]
        pool_feats = s[prune]
        pool_covs = cov_sum[prune]

    if pool_keys.numel():
        survivors_k.append(pool_keys)
        survivors_f.append(pool_feats)
        survivors_c.append(pool_covs)

    sk = torch.cat(survivors_k) if survivors_k else torch.empty(0, dtype=torch.int64, device=device)
    sf = torch.cat(survivors_f) if survivors_f else torch.empty(0, FEAT_DIM, device=device)
    sc = torch.cat(survivors_c) if survivors_c else torch.empty(0, 3, 3, device=device)
    return PrunedOctree(
        keys=sk, feats=sf, covs=sc,
        original_keys=original_keys, original_level=level,
        ratio=n_original / max(sk.numel(), 1),
    )


def decode_to_fine(pruned: PrunedOctree) -> torch.Tensor:
    """Reconstruct every original fine token from its surviving ancestor.

    v_fine = 2^d * v_ancestor - (coord_fine - 2^d * coord_ancestor), clamped.
    Non-geometry channels broadcast unchanged (lossy only in vertex position).
    """
    device = pruned.original_keys.device
    keys = pruned.original_keys
    N = keys.numel()
    cell_part, sheet = token_split(keys)
    coords, _ = morton_decode(cell_part)
    level = pruned.original_level
    index = MortonIndex(pruned.keys)

    out = torch.zeros(N, FEAT_DIM, dtype=pruned.feats.dtype, device=device)
    unresolved = torch.ones(N, dtype=torch.bool, device=device)
    for d in range(0, level):
        if not unresolved.any():
            break
        anc_coords = coords >> d
        anc_key = token_key(morton_encode(anc_coords, level - d), sheet)
        pos = index.query(anc_key)
        found = unresolved & (pos >= 0)
        if found.any():
            f_anc = pruned.feats[pos[found]]
            scale = float(1 << d)
            v = scale * f_anc[:, :3] - (coords[found] - scale * anc_coords[found]).to(f_anc.dtype)
            out[found, :3] = v.clamp(0.0, 1.0)
            out[found, 3:] = f_anc[:, 3:]
            unresolved &= ~found
    if unresolved.any():
        raise RuntimeError(f"{int(unresolved.sum())} tokens lost a surviving ancestor")
    return out