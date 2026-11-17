"""Sinkhorn-Knopp Mini-Batch Optimal Transport Pairing (roadfinal.md §3.3, §7 Phase 5).

Pairs noise samples Z_0 and data samples Z_1 in a mini-batch via 15-step
log-domain Sinkhorn-Knopp optimal transport, straightening flow trajectories
for 1-NFE / 2-NFE MeanFlow convergence.
"""

from __future__ import annotations

import torch


def sinkhorn_knopp_ot(
    cost_matrix: torch.Tensor,
    reg: float = 0.05,
    num_iters: int = 15,
) -> torch.Tensor:
    """Compute doubly-stochastic OT plan pi* using log-stabilized Sinkhorn-Knopp.

    Args:
        cost_matrix: [B, B] pairwise transport cost matrix (non-negative).
        reg: Entropic regularization epsilon.
        num_iters: Number of Sinkhorn scaling iterations (default 15 per spec).

    Returns:
        pi: [B, B] transport plan with rows/cols summing to 1/B.
    """
    b = cost_matrix.shape[0]
    if b <= 1:
        return torch.ones_like(cost_matrix)

    # Scale cost by median/max to keep exp(-C/reg) well-conditioned
    scale = cost_matrix.detach().mean().clamp(min=1e-6)
    c_norm = cost_matrix / scale

    log_k = -c_norm / max(reg, 1e-4)
    log_a = torch.full((b,), -torch.log(torch.tensor(float(b), device=cost_matrix.device)),
                       device=cost_matrix.device, dtype=cost_matrix.dtype)
    u = torch.zeros(b, device=cost_matrix.device, dtype=cost_matrix.dtype)
    v = torch.zeros(b, device=cost_matrix.device, dtype=cost_matrix.dtype)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
        v = log_a - torch.logsumexp(log_k + u.unsqueeze(1), dim=0)

    log_pi = u.unsqueeze(1) + log_k + v.unsqueeze(0)
    return torch.exp(log_pi)


@torch.no_grad()
def ot_pair_batch(
    z0: torch.Tensor,
    z1: torch.Tensor,
    reg: float = 0.05,
    num_iters: int = 15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder z0 to match z1 via mini-batch Sinkhorn-Knopp OT.

    Args:
        z0: Noise tensor [B, ...].
        z1: Data tensor [B, ...].
        reg: Entropic regularization.
        num_iters: Sinkhorn iterations.

    Returns:
        (z0_paired, perm_indices, pi):
          z0_paired: z0 reordered to minimize transport cost to z1.
          perm_indices: [B] integer indices into original z0.
          pi: [B, B] Sinkhorn transport plan.
    """
    b = z0.shape[0]
    if b <= 1:
        idx = torch.arange(b, device=z0.device)
        return z0, idx, torch.ones((b, b), device=z0.device, dtype=z0.dtype)

    x0 = z0.reshape(b, -1).float()
    x1 = z1.reshape(b, -1).float()
    dim = max(1, x0.shape[1])

    # Pairwise squared Euclidean distance normalized by dimension
    cost = torch.cdist(x0, x1, p=2.0).pow(2) / float(dim)
    pi = sinkhorn_knopp_ot(cost, reg=reg, num_iters=num_iters)

    # Extract greedy-max permutation without duplicate assignments
    # (for small/medium B, iterative argmax over pi gives exact bijection)
    perm = torch.empty(b, dtype=torch.long, device=z0.device)
    pi_work = pi.clone()
    for j in range(b):
        # Assign best remaining z0 row to data column j
        i = torch.argmax(pi_work[:, j])
        perm[j] = i
        pi_work[i, :] = -1.0

    return z0[perm], perm, pi
