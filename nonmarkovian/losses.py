"""Training losses for discrete DNA diffusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence_logits_onehot(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int = 4,
) -> torch.Tensor:
    """
    Per-token ``KL(p || q)`` with ``p = one_hot(target)``, ``q = softmax(logits)``.

    Shape: ``logits`` ``[B, L, num_classes]``, ``target`` long ``[B, L]`` in ``0 .. num_classes-1``.

    For hard labels this equals cross-entropy ``-log q_y`` (``H(p)=0`` for a delta).
    Implemented as ``-sum_c p_c log q_c = -log q_y`` for one-hot ``p``.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    tg = target.long().clamp(max=num_classes - 1)
    oh = F.one_hot(tg, num_classes=num_classes).to(dtype=log_probs.dtype)
    return -(oh * log_probs).sum(dim=-1)
