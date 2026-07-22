"""MDLM (masked / absorbing-state) forward corruption — drop-in analog of ``forward.py``.

Where ``forward.py`` corrupts via multi-class **Bernoulli** flips on a simplex, this module
corrupts via **absorbing-mask** noising: each token independently becomes ``[M]`` (``MASK_IDX``)
with probability ``mask_prob(t)``. This is the forward process of Masked Diffusion Language
Models (MDLM, Sahoo et al. 2024) adapted to the SLM-style framework used here.

Design choices (so the Bernoulli-vs-MDLM ablation is clean):

* **Time convention** is identical to ``forward.py`` / SLM: ``t in (0, 1]``, with ``t -> 1`` the
  fully-noised limit (everything masked) and ``t -> 0`` clean.
* **Loglinear schedule** reuses the same ``C**t`` functional form as the Bernoulli noise level,
  so the *amount* of corruption vs. ``t`` matches the Bernoulli code's curve.
* **Representation** is unchanged: masked tokens are fed to the (identical) models via
  ``tokens_to_four_channel_simplex`` (mask -> uniform 1/4), so no model edits are needed.

The MDLM NELBO per-time weight (``mdlm_loss_weight``) and the multi-view sampler
(``sample_all_views_mask``) mirror ``sample_all_views_bernoulli``'s API so the routed
non-Markovian trainer/sampler can swap corruption with no structural change.
"""

from __future__ import annotations

import math

import torch

from nonmarkovian.slm_cnn import tokens_to_four_channel_simplex
from nonmarkovian.vocab import MASK_IDX


def mdlm_mask_prob(
    t: torch.Tensor,
    *,
    num_classes: int = 4,
    scheduler: str = "loglinear",
) -> torch.Tensor:
    """Masking probability ``1 - alpha_t`` at continuous time ``t`` (any shape).

    - ``loglinear``: ``(C**t - 1) / (C - 1)`` — matches the Bernoulli noise-level curve.
    - ``linear``:    ``t`` (canonical MDLM ``alpha_t = 1 - t``).

    Both are ``0`` at ``t=0`` and ``1`` at ``t=1`` and monotonically increasing.
    """
    if scheduler == "loglinear":
        C = float(num_classes)
        ln_c = torch.log(torch.tensor(C, device=t.device, dtype=torch.float32))
        p = (torch.exp(ln_c * t) - 1.0) / (C - 1.0)
    elif scheduler == "linear":
        p = t.to(dtype=torch.float32).clone()
    else:
        raise ValueError(f"Unknown MDLM scheduler: {scheduler!r}")
    return p.clamp(min=0.0, max=1.0)


def mdlm_alpha(
    t: torch.Tensor,
    *,
    num_classes: int = 4,
    scheduler: str = "loglinear",
) -> torch.Tensor:
    """Survival probability ``alpha_t = 1 - mask_prob(t)`` (prob a token stays unmasked)."""
    return 1.0 - mdlm_mask_prob(t, num_classes=num_classes, scheduler=scheduler)


def mdlm_loss_weight(
    t: torch.Tensor,
    *,
    num_classes: int = 4,
    scheduler: str = "loglinear",
    eps: float = 1e-6,
) -> torch.Tensor:
    """MDLM NELBO per-time weight ``w(t) = -alpha'_t / (1 - alpha_t) = p'(t) / p(t)``.

    With mask-prob ``p(t)``:
      - ``loglinear``: ``p = (C**t - 1)/(C - 1)``, ``p' = C**t ln C/(C - 1)``
                       ``=> w = C**t ln C / (C**t - 1)``
      - ``linear``:    ``p = t``, ``p' = 1`` ``=> w = 1 / t``

    Both ``-> 1/t`` as ``t -> 0`` (the canonical MDLM weight). The loss is the
    Monte-Carlo estimate ``E_{t~U(0,1]}[ w(t) * sum_{masked} CE ]`` of the
    continuous-time NELBO ``integral_0^1 w(t) * sum_{masked} CE dt``.
    """
    t = t.to(dtype=torch.float32)
    if scheduler == "loglinear":
        C = float(num_classes)
        ln_c = math.log(C)
        c_t = torch.exp(torch.log(torch.tensor(C, device=t.device, dtype=torch.float32)) * t)
        w = (c_t * ln_c) / (c_t - 1.0).clamp(min=eps)
    elif scheduler == "linear":
        w = 1.0 / t.clamp(min=eps)
    else:
        raise ValueError(f"Unknown MDLM scheduler: {scheduler!r}")
    return w


def corrupt_mask(
    x0: torch.Tensor,
    t: torch.Tensor,
    *,
    num_classes: int = 4,
    scheduler: str = "loglinear",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Absorbing-mask corruption of a single view (analog of ``corrupt_sequence_bernoulli``).

    Args:
        x0: ``[B, L]`` token ids in ``0..num_classes-1``.
        t:  ``[B, 1]`` (or broadcastable) normalized timesteps in ``(0, 1]``.
    Returns:
        ``x_t``: ``[B, L]`` long ids, masked positions set to ``MASK_IDX``.
    """
    p = mdlm_mask_prob(t, num_classes=num_classes, scheduler=scheduler)  # [B, 1]
    if generator is None:
        u = torch.rand(x0.shape, device=x0.device, dtype=torch.float32)
    else:
        u = torch.rand(x0.shape, device=x0.device, dtype=torch.float32, generator=generator)
    mask = u < p  # broadcast [B, 1] over L
    out = x0.clone()
    out[mask] = MASK_IDX
    return out


def sample_all_views_mask(
    x0: torch.Tensor,
    num_timesteps: int,
    *,
    t_start: int | None = None,
    scheduler: str = "loglinear",
    generator: torch.Generator | None = None,
    num_classes: int = 4,
    corruption_mode: str = "independent",
    return_simplex: bool = True,
) -> torch.Tensor:
    """Masked-diffusion analog of ``sample_all_views_bernoulli``.

    Builds views ``tau in [t_start, T-1]`` (default all), each masked at ``t = (tau + 1) / T``.

    Returns:
        ``return_simplex=True``  (default): ``[B, K, L, 4]`` 4-channel simplex (mask -> uniform
            1/4), ready for the router / CNN (which require 4-channel inputs).
        ``return_simplex=False``: ``[B, K, L]`` long token ids (mask = ``MASK_IDX``); useful for
            recovering which positions are masked in the current view (for the MDLM loss).

    ``corruption_mode``:
        - ``"independent"`` *(default)*: fresh mask draw per timestep — masking at different
          times is i.i.d. given ``x_0``.
        - ``"trajectory"``: one ``u ~ U[0,1]`` per ``(B, L)`` shared across timesteps. Because
          ``mask_prob`` increases with ``tau``, once a position is masked at some ``tau`` it stays
          masked for all noisier ``tau`` — the monotone support constraint (mirrors the Bernoulli
          ``trajectory`` mode), so the current (least-noisy) view's unmasked support contains
          every candidate view's support.
    """
    T = int(num_timesteps)
    if T < 1:
        raise ValueError("num_timesteps must be >= 1")
    tau_begin = 0 if t_start is None else int(t_start)
    if not (0 <= tau_begin < T):
        raise ValueError(f"t_start must be in [0, {T - 1}]")
    if corruption_mode not in ("independent", "trajectory"):
        raise ValueError(
            f"corruption_mode must be 'independent' or 'trajectory', got {corruption_mode!r}"
        )

    B, L = x0.shape
    K = T - tau_begin
    device = x0.device

    taus = torch.arange(tau_begin, T, device=device, dtype=torch.float32)
    t_cont = (taus + 1.0) / float(T)  # [K]
    p = mdlm_mask_prob(t_cont, num_classes=num_classes, scheduler=scheduler).view(1, K, 1)  # [1,K,1]

    if corruption_mode == "independent":
        shape = (B, K, L)
        if generator is None:
            u = torch.rand(shape, device=device, dtype=torch.float32)
        else:
            u = torch.rand(shape, device=device, dtype=torch.float32, generator=generator)
    else:
        # "trajectory": one u per (B, L) shared across all timesteps. Since p is monotone
        # increasing, u < p_{tau} for the first tau and stays true for larger tau.
        shape = (B, 1, L)
        if generator is None:
            u = torch.rand(shape, device=device, dtype=torch.float32)
        else:
            u = torch.rand(shape, device=device, dtype=torch.float32, generator=generator)

    mask = u < p  # [B, K, L] (u broadcasts over K in trajectory mode)
    views_ids = x0.unsqueeze(1).expand(B, K, L).clone()
    views_ids[mask] = MASK_IDX

    if not return_simplex:
        return views_ids
    return tokens_to_four_channel_simplex(views_ids.reshape(B * K, L)).view(B, K, L, 4)
