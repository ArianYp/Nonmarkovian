"""Baseline discrete diffusion with DiT architecture: DDiTBlocks with AdaLN, rotary embeddings.

Matches the paper's DIT backbone — every block receives timestep conditioning via
adaptive layer-norm (AdaLN) modulation. Total blocks = ``dec_layers``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nonmarkovian.dit import (
    DDiTBlock,
    DDitFinalLayer,
    EmbeddingLayer,
    LabelEmbedder,
    Rotary,
    TimestepEmbedder,
    amp_context,
)
from nonmarkovian.model import ActivityAuxHead
from nonmarkovian.vocab import VOCAB_SIZE


class _T0EncoderProxy:
    """Runs the DiT backbone with t=0 conditioning for FBD embeddings."""

    def __init__(self, model: "DiscreteDenoiser"):
        self._model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = self._model
        t = x.new_zeros(x.shape[0], dtype=torch.long)
        c = F.silu(m.sigma_map(t))
        h = m.vocab_embed(x)
        rot = m.rotary_emb(h)
        with amp_context(x.device):
            for block in m.blocks:
                h = block(h, rot, c)
        return h


class DiscreteDenoiser(nn.Module):
    """DiT-based absorbing-mask diffusion denoiser.

    Architecture (per the baseline paper):
      1. EmbeddingLayer(d_model, VOCAB_SIZE)  — Kaiming-uniform init
      2. TimestepEmbedder → SiLU → conditioning vector c
      3. Optional LabelEmbedder added to c
      4. Rotary positional embeddings (applied inside each block)
      5. n_blocks × DDiTBlock with AdaLN from c, flash attention (or SDPA fallback)
      6. DDitFinalLayer with AdaLN → 4-way logits (ACGT)
    """

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dec_layers: int,
        dim_ff: int,
        dropout: float,
        max_len: int,
        num_timesteps: int,
        num_labels: int | None = None,
        label_dim: int | None = None,
        cond_dim: int | None = None,
        time_freq_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_timesteps = num_timesteps
        self.max_len = max_len

        n_blocks = dec_layers
        if cond_dim is None:
            cond_dim = d_model
        self.cond_dim = cond_dim

        self.vocab_embed = EmbeddingLayer(d_model, VOCAB_SIZE)
        self.sigma_map = TimestepEmbedder(cond_dim, frequency_embedding_size=time_freq_dim)
        self.rotary_emb = Rotary(d_model // nhead)

        self.blocks = nn.ModuleList(
            [DDiTBlock(d_model, nhead, cond_dim, dim_ff=dim_ff, dropout=dropout) for _ in range(n_blocks)]
        )
        self.output_layer = DDitFinalLayer(d_model, 4, cond_dim)

        self.num_labels = num_labels
        if num_labels is not None and num_labels > 0:
            self.label_embed = LabelEmbedder(num_labels, cond_dim)
        else:
            self.label_embed = None

        self._encoder_proxy = _T0EncoderProxy(self)

    @property
    def encoder(self):
        """Callable ``encoder(x) -> [B, L, d]`` for FBD embeddings (runs backbone with t=0)."""
        return self._encoder_proxy

    def forward(
        self,
        x_t: torch.Tensor,
        t_idx: torch.Tensor | int,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_t: [B, L] token ids (includes MASK).
        t_idx: [B] or scalar int in [0, num_timesteps).
        Returns logits [B, L, 4], hidden [B, L, d].
        """
        B = x_t.shape[0]
        device = x_t.device
        if isinstance(t_idx, int):
            t_b = torch.full((B,), t_idx, device=device, dtype=torch.long)
        else:
            t_b = t_idx.long().view(B)

        x = self.vocab_embed(x_t)
        c = F.silu(self.sigma_map(t_b))

        if self.label_embed is not None and labels is not None:
            c = c + self.label_embed(labels)

        rotary_cos_sin = self.rotary_emb(x)

        with amp_context(device):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c)
            h_dec = x
            logits = self.output_layer(x, c)

        return logits, h_dec


__all__ = ["DiscreteDenoiser", "ActivityAuxHead"]
