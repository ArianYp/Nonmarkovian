"""Boltzmann router + single DiT denoiser for routed non-Markovian diffusion.

One DiT stack (AdaLN, rotary) matches the baseline: it only runs on the routed token
context.  The router uses shared **token embeddings** (no extra Transformer) for
h_t, g_k and per-position mixing — same EmbeddingLayer weights the DiT uses at its input.
"""

from __future__ import annotations

import math

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
from nonmarkovian.vocab import VOCAB_SIZE


class _EncoderCallable:
    """Thin wrapper so ``model.encoder(x)`` works for FBD without nn.Module registration."""

    def __init__(self, encode_fn):
        self._fn = encode_fn

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x)


class RoutedDenoiser(nn.Module):
    """
    At reverse step ``t_start`` (0 … T−1):
    - ``h_t`` = mean-pool of **token embeddings** of ``x_{t_start}``.
    - Candidates ``k ∈ {t_start+1, …, T−1}``: ``g_k`` = mean-pool of embeddings of ``x_k``.
    - Compatibility ``e_k = (h_t^T W g_k) / √d`` over **all** future candidates. ``π =`` Gumbel–Softmax(``e``, τ) in training and ``softmax(e/τ)`` at eval — fully differentiable; ``ctx = z_t + Σ_k π_k z_k`` (no top-k / no straight-through).
    - **Single DiT**: ``ctx`` → DDiTBlocks conditioned on ``t_start`` (+ label) → logits.

    Total depth: ``dec_layers`` DDiT blocks.
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
        router_tau: float = 1.0,
        router_k: int = 1,  # unused; kept for checkpoint / CLI compatibility
        time_freq_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_timesteps = num_timesteps
        self.max_len = max_len
        self.router_tau = float(router_tau)
        self._inv_sqrt_d = 1.0 / math.sqrt(float(d_model))

        if cond_dim is None:
            cond_dim = d_model
        self.cond_dim = cond_dim

        n_blocks = dec_layers

        # ---- shared input embedding (router + DiT input) ----
        self.vocab_embed = EmbeddingLayer(d_model, VOCAB_SIZE)

        # ---- Boltzmann router ----
        self.W_phi = nn.Linear(d_model, d_model, bias=False)

        # ---- single DiT backbone (denoising only) ----
        self.sigma_map = TimestepEmbedder(cond_dim, frequency_embedding_size=time_freq_dim)
        self.rotary = Rotary(d_model // nhead)
        self.num_labels = num_labels
        if num_labels is not None and num_labels > 0:
            self.label_embed = LabelEmbedder(num_labels, cond_dim)
        else:
            self.label_embed = None

        self.blocks = nn.ModuleList(
            [DDiTBlock(d_model, nhead, cond_dim, dim_ff=dim_ff, dropout=dropout) for _ in range(n_blocks)]
        )
        self.output_layer = DDitFinalLayer(d_model, 4, cond_dim)

        self._enc_callable = _EncoderCallable(self._encode_tokens_t0)

    # ----- embedding (router path; no Transformer) ---------------------------

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Token ids [B, L] -> embeddings [B, L, d]."""
        return self.vocab_embed(x)

    # ----- single DiT (full depth) -----------------------------------------

    def _dit_features(self, x: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """Run DiT blocks on token ids with conditioning timestep t_idx [B]. Returns [B, L, d]."""
        h = self.vocab_embed(x)
        c = F.silu(self.sigma_map(t_idx))
        rot = self.rotary(h)
        with amp_context(x.device):
            for block in self.blocks:
                h = block(h, rot, c)
        return h

    @property
    def encoder(self):
        """FBD: DiT hidden states at diffusion index0, no output head."""
        return self._enc_callable

    def _encode_tokens_t0(self, x: torch.Tensor) -> torch.Tensor:
        t0 = x.new_zeros(x.shape[0], dtype=torch.long)
        return self._dit_features(x, t0)

    def encode_all_views(self, x_views: torch.Tensor) -> torch.Tensor:
        """[B, T, L] -> [B, T, L, d] via **embeddings only** (cheap; for debugging)."""
        B, T, L = x_views.shape
        return self._embed(x_views.reshape(B * T, L)).view(B, T, L, -1)

    def _embed_current_and_candidates(
        self, x_views: torch.Tensor, t_start: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        B, T, L = x_views.shape
        device = x_views.device
        x_t = x_views[:, t_start, :]
        z_t = self._embed(x_t)

        K = T - t_start - 1
        if K <= 0:
            return z_t, None, None

        rows = [self._embed(x_views[:, k_abs, :]) for k_abs in range(t_start + 1, T)]
        z_cand = torch.stack(rows, dim=1)
        taus_cand = torch.tensor(list(range(t_start + 1, T)), device=device, dtype=torch.long)
        return z_t, z_cand, taus_cand

    def _compatibility_scores(self, h_t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h_w = self.W_phi(h_t)
        return (h_w.unsqueeze(1) * g).sum(dim=-1) * self._inv_sqrt_d

    def _router_forward(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tau = max(self.router_tau, 1e-6)
        pi_soft = torch.softmax(e / tau, dim=-1)
        if self.training:
            pi = F.gumbel_softmax(e, tau=tau, dim=-1, hard=False)
        else:
            pi = pi_soft
        return pi, pi_soft, pi

    def _load_balance_loss(self, e: torch.Tensor, pi_soft: torch.Tensor) -> torch.Tensor:
        B, K = e.shape
        if K == 0:
            return e.new_tensor(0.0)
        k_hard = e.argmax(dim=-1)
        f = F.one_hot(k_hard, num_classes=K).to(dtype=e.dtype).mean(dim=0)
        bar_pi = pi_soft.mean(dim=0)
        return (float(K) * (f * bar_pi).sum()).to(e.dtype)

    def forward(
        self,
        x_views: torch.Tensor,
        t_start: int,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, L = x_views.shape
        device = x_views.device
        if not (0 <= t_start < T):
            raise ValueError("t_start out of range")

        z_t, z_cand, _ = self._embed_current_and_candidates(x_views, t_start)

        if z_cand is None:
            ctx = z_t
            pi = z_t.new_zeros(B, 0)
            loss_bal = z_t.new_tensor(0.0)
        else:
            h_t = z_t.mean(dim=1)
            g = z_cand.mean(dim=2)
            e = self._compatibility_scores(h_t, g)
            pi_hat, pi_soft, _ = self._router_forward(e)
            loss_bal = self._load_balance_loss(e, pi_soft) if self.training else e.new_tensor(0.0)
            pi_w = pi_hat.view(B, -1, 1, 1)
            ctx_mix = (z_cand * pi_w).sum(dim=1)
            ctx = z_t + ctx_mix
            pi = pi_hat

        t_b = torch.full((B,), int(t_start), device=device, dtype=torch.long)
        c = F.silu(self.sigma_map(t_b))
        if self.label_embed is not None and labels is not None:
            c = c + self.label_embed(labels)

        rot = self.rotary(ctx)
        with amp_context(ctx.device):
            for block in self.blocks:
                ctx = block(ctx, rot, c)
            h_dec = ctx
            logits = self.output_layer(ctx, c)

        return logits, pi, h_dec, loss_bal


class ActivityAuxHead(nn.Module):
    """Optional predictor on mean-pooled denoiser hidden state."""

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, num_classes))

    def forward(self, h_tokens: torch.Tensor) -> torch.Tensor:
        h = h_tokens.mean(dim=1)
        return self.net(h)
