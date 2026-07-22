"""Baseline (non-routed, Markovian) protein discrete diffusion with the SLM ``dit_bfn`` backbone.

This is the protein analogue of :class:`nonmarkovian.simple_model.DiscreteDenoiser` -- the
plain SLM ``new_diff`` denoiser with **no router and no multi-view history**. It denoises a
single corrupted simplex ``x_t`` at one timestep, exactly like ``SLM/slm.py:_forward_new_diffusion``
with ``backbone=dit_bfn``. Use it as the vanilla baseline to compare against the
non-Markovian routed model (:class:`nonmarkovian.model_protein.RoutedDenoiserDiTBFN`).

``forward(x_t, t) -> (logits, None)`` matches the simple-model contract so the simple
training / sampling code is shared.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from nonmarkovian.dit_bfn import BFN_DIT


class DiscreteDenoiserDiTBFN(nn.Module):
    """Plain ``BFN_DIT`` denoiser over a ``vocab_size``-channel simplex (no routing)."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_len: int,
        num_timesteps: int,
        hidden_size: int = 480,
        cond_dim: int = 128,
        n_blocks: int = 12,
        n_heads: int = 8,
        dropout: float = 0.1,
        scale_by_sigma: bool = True,
        embedding_nml: bool = False,
        entropy_condition: bool = False,
        num_labels: int | None = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.num_timesteps = int(num_timesteps)
        self.max_len = int(max_len)
        self.num_labels = num_labels  # UniRef is unconditional; kept for API parity
        cfg = OmegaConf.create(
            {
                "embedding_nml": bool(embedding_nml),
                "entropy_condition": bool(entropy_condition),
                "model": {
                    "hidden_size": int(hidden_size),
                    "cond_dim": int(cond_dim),
                    "n_heads": int(n_heads),
                    "n_blocks": int(n_blocks),
                    "dropout": float(dropout),
                    "scale_by_sigma": bool(scale_by_sigma),
                },
            }
        )
        self.dit = BFN_DIT(cfg, vocab_size=self.vocab_size)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | float | int,
        labels: torch.Tensor | None = None,
    ):
        """``x_t``: ``[B, L, V]`` simplex (or ``[B, L]`` token ids). ``t``: scalar or ``[B]`` in [0, 1].

        Returns ``(logits [B, L, V], None)``.
        """
        if x_t.ndim == 3:
            seq = x_t.to(dtype=torch.float32)  # already a simplex
        else:
            seq = F.one_hot(x_t.long(), num_classes=self.vocab_size).to(dtype=torch.float32)
        B = seq.shape[0]
        device = seq.device
        if isinstance(t, (int, float)):
            sigma = torch.full((B,), float(t), device=device, dtype=torch.float32)
        else:
            sigma = t.to(device=device, dtype=torch.float32).view(-1)
            if sigma.numel() == 1:
                sigma = sigma.expand(B).contiguous()
        logits = self.dit(seq, sigma)  # [B, L, V]
        return logits, None


__all__ = ["DiscreteDenoiserDiTBFN"]
