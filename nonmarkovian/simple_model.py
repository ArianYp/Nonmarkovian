"""Baseline discrete diffusion with DiT architecture: DDiTBlocks with AdaLN, rotary embeddings.

Matches the paper's DIT backbone — every block receives timestep conditioning via
adaptive layer-norm (AdaLN) modulation. Total blocks = ``dec_layers``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nonmarkovian.dit import (
    DDiTBlock,
    DDitFinalLayer,
    LabelEmbedder,
    Rotary,
    TimestepEmbedder,
    amp_context,
)
from nonmarkovian.slm_cnn import CNNModel, timestep_index_to_float, tokens_to_four_channel_simplex


class _T0EncoderProxy:
    """Runs the DiT backbone with t=0 conditioning for FBD embeddings."""

    def __init__(self, model: "DiscreteDenoiser"):
        self._model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = self._model
        t = x.new_zeros(x.shape[0], dtype=torch.long)
        c = F.silu(m.sigma_map(t))
        seq = tokens_to_four_channel_simplex(x)
        h = m.input_proj(seq)
        rot = m.rotary_emb(h)
        with amp_context(x.device):
            for block in m.blocks:
                h = block(h, rot, c)
        return h


class _CNNEncoderProxy:
    """CNN stack at t≈0 for FBD-style pooled embeddings (SLM ``hidden_dim`` = 128)."""

    def __init__(self, model: "DiscreteDenoiserCNN") -> None:
        self._model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = self._model
        seq = tokens_to_four_channel_simplex(x)
        t0 = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        feat_bhl = m.cnn._feature_map_bhl(seq, t0, None)
        return feat_bhl.permute(0, 2, 1).contiguous()


class DiscreteDenoiser(nn.Module):
    """DiT-based absorbing-mask diffusion denoiser.

    Token ids (ACGT + ``[M]``) map to a 4-channel simplex via ``tokens_to_four_channel_simplex``
    (mask → uniform ¼); a linear layer projects to ``d_model`` (no discrete vocab embedding table).
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

        self.input_proj = nn.Linear(4, d_model)
        nn.init.kaiming_uniform_(self.input_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.input_proj.bias)
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
        t_idx: torch.Tensor | float | int,
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
        elif isinstance(t_idx, float):
            t_b = torch.full((B,), t_idx, device=device, dtype=torch.float32)
        else:
            t_b = t_idx.view(B)

        if x_t.ndim == 3:
            seq = x_t.to(dtype=torch.float32)
        else:
            seq = tokens_to_four_channel_simplex(x_t)
        x = self.input_proj(seq)
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


class DiscreteDenoiserCNN(nn.Module):
    """Discrete diffusion denoiser using the same 1D CNN as ``SLM/models/dna_models.py``."""

    def __init__(
        self,
        *,
        d_model: int,
        max_len: int,
        num_timesteps: int,
        num_labels: int | None = None,
        num_cnn_stacks: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_timesteps = num_timesteps
        self.max_len = max_len

        cnn_num_cls = num_labels if num_labels is not None and num_labels > 0 else 1
        self.num_labels = num_labels
        self.cnn = CNNModel(4, 81, num_cnn_stacks, classifier=False, max_len=max_len)

        self._encoder_proxy = _CNNEncoderProxy(self)

    @property
    def encoder(self):
        return self._encoder_proxy

    def forward(
        self,
        x_t: torch.Tensor,
        t_idx: torch.Tensor | float | int,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B = x_t.shape[0]
        device = x_t.device
        if isinstance(t_idx, int):
            t_b = torch.full((B,), t_idx, device=device, dtype=torch.float32)
        elif isinstance(t_idx, float):
            t_b = torch.full((B,), t_idx, device=device, dtype=torch.float32)
        else:
            t_b = t_idx.to(device=device).view(B)

        if x_t.ndim == 3:
            seq = x_t.to(dtype=torch.float32)
        else:
            seq = tokens_to_four_channel_simplex(x_t)
        if torch.is_floating_point(t_b):
            t_cont = t_b.to(dtype=torch.float32).clamp(min=0.0, max=1.0)
        else:
            t_cont = timestep_index_to_float(t_b.long(), self.num_timesteps)
        cls_inp = None
        if self.num_labels is not None and self.num_labels > 0 and labels is not None:
            cls_inp = labels
        logits = self.cnn(seq, t_cont, cls_inp)
        return logits, None


__all__ = ["DiscreteDenoiser", "DiscreteDenoiserCNN"]
