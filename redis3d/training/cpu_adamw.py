"""Pinned 32GB DDR5 CPU AdamW offload optimizer (roadfinal.md §4.2, §7 Phase 4).

Keeps optimizer states (first and second moments m, v) in pinned host RAM
so that GPU training optimizer states take 0 MB VRAM, allowing full BF16
model training within the 8 GB RTX 5060 budget (< 5.2 GB total VRAM).
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


class CPUAdamW(Optimizer):
    """AdamW with first/second moments offloaded to pinned CPU host memory.
    
    GPU VRAM overhead for optimizer states is strictly 0 MB.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CPUAdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization in pinned CPU memory (0 MB GPU VRAM!)
                if len(state) == 0:
                    state["step"] = 0
                    # Pinned CPU memory for fast async DMA transfer
                    state["exp_avg"] = torch.zeros(
                        p.shape, dtype=torch.float32, device="cpu", pin_memory=torch.cuda.is_available()
                    )
                    state["exp_avg_sq"] = torch.zeros(
                        p.shape, dtype=torch.float32, device="cpu", pin_memory=torch.cuda.is_available()
                    )

                state["step"] += 1
                step = state["step"]

                # Move grad to CPU float32 synchronously for race-free moment accumulation
                grad_cpu = grad.detach().to("cpu", dtype=torch.float32)
                p_cpu = p.detach().to("cpu", dtype=torch.float32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Perform weight decay
                if wd != 0.0:
                    p_cpu.mul_(1.0 - lr * wd)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad_cpu, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1.0 - beta2)

                # Bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                p_cpu.addcdiv_(exp_avg, denom, value=-step_size)

                # Copy updated weights back to GPU parameter
                p.copy_(p_cpu.to(p.device, dtype=p.dtype))

        return loss
