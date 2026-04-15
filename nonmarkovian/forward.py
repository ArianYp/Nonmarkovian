"""Non-Markovian forward corruption: independent absorbing [M] noise per timestep."""

from __future__ import annotations

import torch

from nonmarkovian.vocab import MASK_IDX


def cosine_alpha_schedule(num_steps: int, device: torch.device | None = None) -> torch.Tensor:
    """Monotone α_t in [0, 1] with α_1 <= ... <= α_T (t = 1..T)."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    # t linear in [0,1], α = 0.02 + 0.98 * (1 - cos(π t / 2))  gives smooth increase
    t = torch.linspace(0.0, 1.0, num_steps, device=device)
    alpha = 0.02 + 0.98 * (1.0 - torch.cos(0.5 * torch.pi * t))
    alpha = torch.clamp(alpha, min=0.0, max=1.0)
    for i in range(1, len(alpha)):
        if alpha[i] < alpha[i - 1]:
            alpha[i] = alpha[i - 1]
    return alpha


def corrupt_sequence(x0: torch.Tensor, alpha: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Per-position mask with prob alpha; x0 is [B, L] long in 0..3."""
    if generator is None:
        u = torch.rand(x0.shape, device=x0.device, dtype=torch.float32)
    else:
        u = torch.rand(x0.shape, device=x0.device, dtype=torch.float32, generator=generator)
    mask = u < float(alpha)
    out = x0.clone()
    out[mask] = MASK_IDX
    return out


def sample_all_views(
    x0: torch.Tensor,
    alphas: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample x_1..x_T independently from x_0. Returns [B, T, L]."""
    B, L = x0.shape
    T = int(alphas.shape[0])
    views = []
    for t in range(T):
        views.append(corrupt_sequence(x0, float(alphas[t].item()), generator=generator))
    return torch.stack(views, dim=1)


def transition_from_predicted_x0(
    x0_pred: torch.Tensor,
    alpha_prev: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """q(x_{t-1} | x0): keep nucleotide w.p. 1-α, mask w.p. α."""
    return corrupt_sequence(x0_pred, alpha_prev, generator=generator)
