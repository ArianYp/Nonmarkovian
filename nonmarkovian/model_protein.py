"""Non-Markovian routed denoiser with the SLM ``dit_bfn`` backbone (protein task).

Identical routing machinery to :class:`nonmarkovian.model.RoutedDenoiserCNN`, but

- it operates on ``vocab_size``-channel simplices (protein alphabet, 31) instead of
  the 4-channel DNA simplex, and
- the denoiser is SLM's :class:`nonmarkovian.dit_bfn.BFN_DIT` instead of ``CNNModel``.

The Boltzmann router scores each candidate (noisier) view against the current view,
(Gumbel-)softmax-selects a history mix, blends it with the current view, renormalises
to a simplex ``seq_in``, and feeds ``(seq_in, sigma=t_cont)`` to ``BFN_DIT`` -- exactly
the SLM ``new_diff`` input contract (a ``[B, L, V]`` simplex + a per-sample time), but
with the non-Markovian history context mixed in.

``forward`` returns the same 5-tuple as ``RoutedDenoiserCNN`` so the training / sampling
code is shared: ``(logits, pi, h_dec=None, loss_bal, seq_in)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from nonmarkovian.dit_bfn import BFN_DIT
from nonmarkovian.model import _ste_hard_threshold  # shared straight-through threshold


class RoutedDenoiserDiTBFN(nn.Module):
    """Boltzmann router + ``BFN_DIT`` denoiser over a ``vocab_size``-channel simplex."""

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
        router_tau: float = 1.0,
        router_k: int = 1,
        router_conv_kernel: int = 1,
        router_out_channels: int = 128,
        num_labels: int | None = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.num_timesteps = int(num_timesteps)
        self.max_len = int(max_len)
        self.router_tau = float(router_tau)
        self.router_k = int(router_k)
        self.num_labels = num_labels  # UniRef is unconditional; kept for API parity
        self.ctx_mix_eps: float = 1e-4

        rk = int(router_conv_kernel)
        if rk < 1 or rk % 2 == 0:
            raise ValueError("router_conv_kernel must be a positive odd int")
        self.router_conv_kernel = rk
        self.router_out_channels = int(router_out_channels)
        pad = rk // 2
        self.W_phi = nn.Conv1d(self.vocab_size, self.router_out_channels, kernel_size=rk, padding=pad, bias=False)
        self.W_cur = nn.Conv1d(self.vocab_size, self.router_out_channels, kernel_size=rk, padding=pad, bias=False)

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

    # -- router internals (generalised from RoutedDenoiserCNN) ------------------
    def _embed_current_and_candidates(self, x_views: torch.Tensor, t_start: int):
        if x_views.ndim != 4:
            raise ValueError("x_views must be [B, T, L, V]")
        B, T, L, V = x_views.shape
        device = x_views.device
        z_t = x_views[:, t_start].to(dtype=torch.float32)  # [B, L, V]
        K = T - t_start - 1
        if K <= 0:
            return z_t, None, None
        z_cand = x_views[:, t_start + 1 : T].to(dtype=torch.float32)  # [B, K, L, V]
        taus_cand = torch.arange(t_start + 1, T, device=device, dtype=torch.long)
        return z_t, z_cand, taus_cand

    def _compatibility_scores(self, z_t: torch.Tensor, z_cand: torch.Tensor) -> torch.Tensor:
        """``z_t`` ``[B, L, V]``, ``z_cand`` ``[B, K, L, V]`` -> ``[B, K]`` dot similarity."""
        B, Kc, L, V = z_cand.shape
        if V != self.vocab_size:
            raise ValueError(f"expected last dim {self.vocab_size}, got {V}")
        if z_t.shape[1] != L:
            raise ValueError("z_t / candidate length mismatch")
        c_out = self.router_out_channels
        inv_sqrt = 1.0 / math.sqrt(float(L * c_out))
        if self.router_conv_kernel == 1:
            # kernel-1 fast path: <W_cur z_t, W_phi z_cand> = z_t^T (W_cur^T W_phi) z_cand
            wc = self.W_cur.weight.squeeze(-1)  # [c_out, V]
            wp = self.W_phi.weight.squeeze(-1)  # [c_out, V]
            M = wc.transpose(0, 1) @ wp  # [V, V]
            zt_proj = torch.einsum("bli,ij->blj", z_t, M)
            return torch.einsum("blj,bklj->bk", zt_proj, z_cand) * inv_sqrt
        h_cur = self.W_cur(z_t.transpose(1, 2).contiguous())  # [B, c_out, L]
        z_ck = z_cand.reshape(B * Kc, L, V).transpose(1, 2).contiguous()
        h_cand = self.W_phi(z_ck).view(B, Kc, c_out, L)
        return (h_cur[:, None, :, :] * h_cand).sum(dim=(2, 3)) * inv_sqrt

    def _router_forward(self, e: torch.Tensor):
        tau = max(self.router_tau, 1e-6)
        pi_soft = torch.softmax(e / tau, dim=-1)
        if self.training:
            pi = F.gumbel_softmax(e, tau=tau, dim=-1, hard=False)
        else:
            pi = pi_soft
        return pi, pi_soft

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
        t_cond: float | int | None = None,
        t_start_abs: int | None = None,
    ):
        if x_views.ndim != 4:
            raise ValueError("x_views must be [B, T, L, V]")
        B, T, L, V = x_views.shape
        device = x_views.device
        if not (0 <= t_start < T):
            raise ValueError("t_start out of range")
        t_start_state = int(t_start) if t_start_abs is None else int(t_start_abs)

        z_t, z_cand, _taus = self._embed_current_and_candidates(x_views, t_start)

        if z_cand is None:
            ctx = z_t
            pi = z_t.new_zeros(B, 0)
            loss_bal = z_t.new_tensor(0.0)
        else:
            e = self._compatibility_scores(z_t, z_cand)
            pi_hat, pi_soft = self._router_forward(e)
            loss_bal = self._load_balance_loss(e, pi_soft) if self.training else e.new_tensor(0.0)
            pi_w = pi_hat.view(B, -1, 1, 1)
            ctx_mix = (z_cand * pi_w).sum(dim=1)  # [B, L, V]
            ctx_mix = _ste_hard_threshold(ctx_mix, float(self.ctx_mix_eps))
            ctx = 1*z_t + 1*ctx_mix
            pi = pi_hat

        seq_in = ctx / ctx.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, L, V] simplex

        # Per-sample diffusion time in [0, 1] for the DiT sigma conditioning.
        if t_cond is None:
            t_val = (float(t_start_state) + 0.5) / float(self.num_timesteps)
        else:
            t_val = float(t_cond)
            
        sigma = torch.full((B,), t_val, device=device, dtype=torch.float32)

        logits = self.dit(seq_in, sigma)  # [B, L, V] raw logits
        # h_dec is None: BFN_DIT exposes no hidden state for an aux head (UniRef is unconditional).
        return logits, pi, None, loss_bal, seq_in
