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
from nonmarkovian.slm_cnn import CNNModel, timestep_index_to_float, tokens_to_four_channel_simplex
from nonmarkovian.vocab import VOCAB_SIZE


def _ste_hard_threshold(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Zero out entries with ``|x| <= eps`` in the forward pass while letting gradients
    flow through unchanged (straight-through estimator).

    Forward: ``x * (|x| > eps)`` (exact zeros on sub-threshold entries).
    Backward: identity w.r.t. ``x`` (gradient passes through every entry).
    """
    if eps <= 0.0:
        return x
    mask = (x.abs() > eps).to(dtype=x.dtype)
    return x - (x * (1.0 - mask)).detach()


def _mask_to_topk_logits(e: torch.Tensor, k: int) -> torch.Tensor:
    """Keep per-row top-k logits and set others to -inf."""
    if e.ndim != 2:
        return e
    num_cands = int(e.shape[-1])
    kk = int(k)
    if num_cands <= 0 or kk <= 0 or kk >= num_cands:
        return e
    top_idx = torch.topk(e, k=kk, dim=-1).indices
    keep = torch.zeros_like(e, dtype=torch.bool)
    keep.scatter_(1, top_idx, True)
    neg_inf = torch.finfo(e.dtype).min
    return e.masked_fill(~keep, neg_inf)


class _EncoderCallable:
    """Thin wrapper so ``model.encoder(x)`` works for FBD without nn.Module registration."""

    def __init__(self, encode_fn):
        self._fn = encode_fn

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x)


class RoutedDenoiser(nn.Module):
    """
    At reverse step ``t_start`` (0 … T−1):
    - ``z_t`` = **token embeddings** of ``x_{t_start}`` (shape ``[B, L, d]``).
    - Candidates ``k ∈ {t_start+1, …, T−1}``: ``z_k`` full-sequence embeddings ``[B, K, L, d]``.
    - Compatibility ``e_k = (1/L) Σ_ℓ ((W z_t^ℓ) · z_k^ℓ) / √d`` — router scores use **all positions**, not sequence means.
    - ``π =`` Gumbel–Softmax(``e``, τ) in training and ``softmax(e/τ)`` at eval; ``ctx = z_t + Σ_k π_k z_k``.
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
        router_tau: float = 0.1,
        router_k: int = 1,  # unused; kept for checkpoint / CLI compatibility
        time_freq_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_timesteps = num_timesteps
        self.max_len = max_len
        self.router_tau = float(router_tau)
        self.router_k = int(router_k)
        self._inv_sqrt_d = 1.0 / math.sqrt(float(d_model))
        self.ctx_mix_eps: float = 1e-4

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

        # Learnable **state-position** embedding: maps each absolute diffusion
        # state index to a ``cond_dim`` vector that is added to the AdaLN
        # conditioning ``c``. The router's π-weighted sum over picked history
        # positions + the current state embedding tells the DiT *which*
        # history positions were routed in.
        self.state_embed = nn.Embedding(int(num_timesteps), cond_dim)
        nn.init.normal_(self.state_embed.weight, std=0.02)
        # Project the same state embedding into the router feature space so
        # state identity affects the compatibility scores *before* routing.
        self.state_router_proj = nn.Linear(cond_dim, d_model, bias=False)

        self._enc_callable = _EncoderCallable(self._encode_tokens_t0)

    # ----- embedding (router path; no Transformer) ---------------------------

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Token ids [B, L] or simplex [B, L, 4] -> embeddings [B, L, d]."""
        if x.ndim == 3:
            # Bernoulli simplex over nucleotides (A,C,G,T); map by expected embedding.
            emb4 = self.vocab_embed.embedding[:4]
            return x.to(dtype=emb4.dtype) @ emb4
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
        """[B, T, L] or [B, T, L, 4] -> [B, T, L, d] via embeddings only."""
        if x_views.ndim == 4:
            B, T, L, _ = x_views.shape
            return self._embed(x_views.reshape(B * T, L, 4)).view(B, T, L, -1)
        B, T, L = x_views.shape
        return self._embed(x_views.reshape(B * T, L)).view(B, T, L, -1)

    def _embed_current_and_candidates(
        self, x_views: torch.Tensor, t_start: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if x_views.ndim == 4:
            B, T, L, _ = x_views.shape
        else:
            B, T, L = x_views.shape
        device = x_views.device
        x_t = x_views[:, t_start]
        z_t = self._embed(x_t)

        K = T - t_start - 1
        if K <= 0:
            return z_t, None, None

        rows = [self._embed(x_views[:, k_abs]) for k_abs in range(t_start + 1, T)]
        z_cand = torch.stack(rows, dim=1)
        taus_cand = torch.tensor(list(range(t_start + 1, T)), device=device, dtype=torch.long)
        return z_t, z_cand, taus_cand

    def _compatibility_scores_full_sequence(
        self,
        z_t: torch.Tensor,
        z_cand: torch.Tensor,
        state_t: torch.Tensor | None = None,
        state_cand: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-position bilinear scores, mean over length. z_t: [B, L, d], z_cand: [B, K, L, d] -> [B, K]."""
        if state_t is not None:
            z_t = z_t + state_t.unsqueeze(1)
        if state_cand is not None:
            z_cand = z_cand + state_cand.unsqueeze(0).unsqueeze(2)
        h_w = self.W_phi(z_t)
        return (h_w.unsqueeze(1) * z_cand).sum(dim=-1).mean(dim=-1) * self._inv_sqrt_d

    def _router_forward(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tau = max(self.router_tau, 1e-6)
        e_masked = _mask_to_topk_logits(e, self.router_k)
        pi_soft = torch.softmax(e_masked / tau, dim=-1)
        if self.training:
            pi = F.gumbel_softmax(e_masked, tau=tau, dim=-1, hard=False)
        else:
            pi = pi_soft
        return pi, pi_soft, pi

    def _load_balance_loss(self, e: torch.Tensor, pi_soft: torch.Tensor) -> torch.Tensor:
        B, K = e.shape
        if K == 0:
            return e.new_tensor(0.0)
        k_hard = e.argmax(dim=-1)
        f = F.one_hot(k_hard, num_classes=K).to(dtype=e.dtype).mean(dim=0)
        pi_soft = F.softmax(e, dim=-1)
        bar_pi = pi_soft.mean(dim=0)
        return (float(K) * (f * bar_pi).sum()).to(e.dtype)

    def forward(
        self,
        x_views: torch.Tensor,
        t_start: int,
        labels: torch.Tensor | None = None,
        t_cond: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if x_views.ndim == 4:
            B, T, L, _ = x_views.shape
        else:
            B, T, L = x_views.shape
        device = x_views.device
        if not (0 <= t_start < T):
            raise ValueError("t_start out of range")

        z_t, z_cand, taus_cand = self._embed_current_and_candidates(x_views, t_start)

        s_cur = self.state_embed(
            torch.full((B,), int(t_start), device=device, dtype=torch.long)
        )
        #s_cur_router = self.state_router_proj(s_cur)
        s_cur_router = None

        if z_cand is None:
            ctx = z_t
            pi = z_t.new_zeros(B, 0)
            loss_bal = z_t.new_tensor(0.0)
            state_cond = s_cur
        else:
            s_cand = self.state_embed(taus_cand)            # [K, cond_dim]
            #s_cand_router = self.state_router_proj(s_cand)  # [K, d_model]
            s_cand_router = None
            e = self._compatibility_scores_full_sequence(
                z_t,
                z_cand,
                state_t=s_cur_router,
                state_cand=s_cand_router,
            )
            pi_hat, pi_soft, _ = self._router_forward(e)
            loss_bal = self._load_balance_loss(e, pi_soft) if self.training else e.new_tensor(0.0)
            pi_w = pi_hat.view(B, -1, 1, 1)
            ctx_mix = (z_cand * pi_w).sum(dim=1)
            pi_l2 = pi_hat.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt().view(B, 1, 1)
            #ctx_mix = ctx_mix / pi_l2
            ctx_mix = _ste_hard_threshold(ctx_mix, float(self.ctx_mix_eps))
            ctx = z_t + ctx_mix
            pi = pi_hat
            # π-weighted sum of per-state embeddings -> [B, cond_dim].
            state_cond = s_cur + pi_hat @ s_cand            # [B, cond_dim]

        if t_cond is None:
            t_b = torch.full((B,), int(t_start), device=device, dtype=torch.long)
        elif isinstance(t_cond, float):
            t_b = torch.full((B,), float(t_cond), device=device, dtype=torch.float32)
        else:
            t_b = torch.full((B,), int(t_cond), device=device, dtype=torch.long)
        c = F.silu(self.sigma_map(t_b)) + state_cond
        if self.label_embed is not None and labels is not None:
            c = c + self.label_embed(labels)

        rot = self.rotary(ctx)
        with amp_context(ctx.device):
            for block in self.blocks:
                ctx = block(ctx, rot, c)
            h_dec = ctx
            logits = self.output_layer(ctx, c)

        # DiT ctx lives in embedding space, not on the 4-simplex, so there's
        # no natural ``seq_in``. Return None and let callers fall back to the
        # raw current-view mask (x_t > 0) when they need a Bernoulli support
        # mask (see sample.py).
        return logits, pi, h_dec, loss_bal, None


class RoutedDenoiserCNN(nn.Module):
    """Boltzmann router + SLM ``CNNModel`` denoiser.

    Routing uses **4-channel** per-base simplex inputs. ``W_cur`` projects the current view and
    ``W_phi`` projects candidate views. Compatibility is dot similarity in conv feature space:
    ``e_k = ⟨ W_cur(z_t), W_phi(z_{cand,k}) ⟩ / √(C_out·L)`` (both maps ``[B,C_out,L]``). Length ``L`` follows
    the batch (no pad-to-``max_len`` here; that was only needed for the old flattened MLP router).
    Mixed ``ctx`` is renormalized to a simplex per position, then passed to ``CNNModel``.
    """

    def __init__(
        self,
        *,
        d_model: int,
        max_len: int,
        num_timesteps: int,
        num_labels: int | None = None,
        router_tau: float = 0.05,
        router_k: int = 1,
        num_cnn_stacks: int = 4,
        router_conv_kernel: int = 5,
        router_out_channels: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_timesteps = num_timesteps
        self.max_len = max_len
        self.router_tau = router_tau
        self.router_k = int(router_k)
        self.ctx_mix_eps: float = 1e-4
        # Debug logging cadence for router sharpness during training.
        # Set to 0 or negative to disable.
        self.router_log_every: int = 100
        self._router_log_step: int = 0
        print("router_tau \n\n\n\n", router_tau, router_k)
        
        self.router_out_channels = 64
        rk = 1
        print("rk", rk)
        if rk < 1 or rk % 2 == 0:
            raise ValueError("router_conv_kernel must be a positive odd int (e.g. 9)")
        self.router_conv_kernel = rk
        pad = rk // 2

        self.W_phi = nn.Conv1d(
            4, self.router_out_channels, kernel_size=rk, padding=pad, bias=False
        )
        self.W_cur = nn.Conv1d(
            4, self.router_out_channels, kernel_size=rk, padding=pad, bias=False
        )

        cnn_num_cls = num_labels if num_labels is not None and num_labels > 0 else 1
        self.num_labels = num_labels
        self.cnn = CNNModel(4, 81, num_cnn_stacks, classifier=False)

        cnn_hidden = int(self.cnn.hidden_dim)
        # Reuse the CNN's own ``time_embedder`` for absolute history-state
        # indices as well; this keeps current-time and selected-state features
        # in the same representation space.
        # Project the resulting state-time features into the router *input* space so
        # state identity affects the selection scores before ``W_phi``.
        self.state_router_proj = nn.Linear(cnn_hidden, 4, bias=False)

        self._enc_callable = _EncoderCallable(self._encode_tokens_t0)

    def _encode_tokens_t0(self, x: torch.Tensor) -> torch.Tensor:
        seq = tokens_to_four_channel_simplex(x)
        t0 = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        feat_bhl = self.cnn._feature_map_bhl(seq, t0, None)
        return feat_bhl.permute(0, 2, 1).contiguous()

    @property
    def encoder(self):
        return self._enc_callable

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.to(dtype=torch.float32)
        return tokens_to_four_channel_simplex(x)

    def _embed_state_times(self, t_idx: torch.Tensor) -> torch.Tensor:
        t_cont = timestep_index_to_float(t_idx, self.num_timesteps)
        return self.cnn.time_embedder(t_cont)

    def encode_all_views(self, x_views: torch.Tensor) -> torch.Tensor:
        if x_views.ndim == 4:
            B, T, L, _ = x_views.shape
            return self._embed(x_views.reshape(B * T, L, 4)).view(B, T, L, -1)
        B, T, L = x_views.shape
        return self._embed(x_views.reshape(B * T, L)).view(B, T, L, -1)

    def _embed_current_and_candidates(
        self, x_views: torch.Tensor, t_start: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if x_views.ndim == 4:
            B, T, L, _ = x_views.shape
        else:
            B, T, L = x_views.shape
        device = x_views.device
        x_t = x_views[:, t_start]
        z_t = self._embed(x_t)

        K = T - t_start - 1
        if K <= 0:
            return z_t, None, None

        rows = [self._embed(x_views[:, k_abs]) for k_abs in range(t_start + 1, T)]
        z_cand = torch.stack(rows, dim=1)
        taus_cand = torch.tensor(list(range(t_start + 1, T)), device=device, dtype=torch.long)
        return z_t, z_cand, taus_cand

    def _compatibility_scores_full_sequence(
        self,
        z_t: torch.Tensor,
        z_cand: torch.Tensor,
        state_t: torch.Tensor | None = None,
        state_cand: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``z_t``: [B, L, 4], ``z_cand``: [B, K, L, 4] → [B, K] via ``W_cur``/``W_phi`` dot similarity."""
        B, Kc, L, four = z_cand.shape
        if four != 4:
            raise ValueError(f"expected last dim 4, got {four}")
        if z_t.shape[1] != L:
            raise ValueError(f"z_t length {z_t.shape[1]} != candidate length {L}")
        if state_t is not None:
            z_t = z_t + state_t.unsqueeze(1)
        if state_cand is not None:
            z_cand = z_cand + state_cand.unsqueeze(0).unsqueeze(2)
        c_out = self.router_out_channels
        inv_sqrt = 1.0 / math.sqrt(float(L * c_out))
        h_cur = self.W_cur(z_t.transpose(1, 2).contiguous())
        z_ck = z_cand.reshape(B * Kc, L, 4).transpose(1, 2).contiguous()
        h_cand = self.W_phi(z_ck).view(B, Kc, c_out, L)
        return (h_cur[:, None, :, :] * h_cand).sum(dim=(2, 3)) * inv_sqrt

    def _router_forward(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tau = max(self.router_tau, 1e-6)
        #print(tau)
        #print(self.router_tau)
        
        #print("e", e)
        #e_masked = _mask_to_topk_logits(e, self.router_k)
        #print(e_masked)
        pi_soft = torch.softmax(e / tau, dim=-1)
        if self.training:
            pi = F.gumbel_softmax(e, tau=tau, dim=-1, hard=False)
        else:
            pi = pi_soft
        if self.training:
            self._router_log_step += 1
            if self.router_log_every > 0 and (self._router_log_step % self.router_log_every == 0):
                with torch.no_grad():
                    peak_soft = pi_soft.max(dim=-1).values.mean().item()
                    peak_sample = pi.max(dim=-1).values.mean().item()
                    entropy_soft = (-(pi_soft * (pi_soft.clamp(min=1e-8).log())).sum(dim=-1)).mean().item()
                    print(
                        "[router]",
                        f"step={self._router_log_step}",
                        f"tau={tau:.6f}",
                        f"peak_soft={peak_soft:.4f}",
                        f"peak_sample={peak_sample:.4f}",
                        f"entropy_soft={entropy_soft:.4f}",
                    )
        #print(pi, pi_soft)
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
        t_cond: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if x_views.ndim == 4:
            B, T, L, _ = x_views.shape
        else:
            B, T, L = x_views.shape
        device = x_views.device
        if not (0 <= t_start < T):
            raise ValueError("t_start out of range")

        z_t, z_cand, taus_cand = self._embed_current_and_candidates(x_views, t_start)

        # Absolute history-state features from the same ``cnn.time_embedder`` used
        # for the current diffusion time conditioning.
        s_cur = self._embed_state_times(
            torch.full((B,), int(t_start), device=device, dtype=torch.long)
        )
        #s_cur_router = self.state_router_proj(s_cur)    # [B, 4]
        s_cur_router = None

        if z_cand is None:
            ctx = z_t
            pi = z_t.new_zeros(B, 0)
            loss_bal = z_t.new_tensor(0.0)
            state_cond = s_cur
        else:
            s_cand = self._embed_state_times(taus_cand)     # [K, H]
            #s_cand_router = self.state_router_proj(s_cand)  # [K, 4]
            s_cand_router = None
            e = self._compatibility_scores_full_sequence(
                z_t,
                z_cand,
                state_t=s_cur_router,
                state_cand=s_cand_router,
            )
            pi_hat, pi_soft, _ = self._router_forward(e)
            loss_bal = self._load_balance_loss(e, pi_soft) if self.training else e.new_tensor(0.0)
            pi_w = pi_hat.view(B, -1, 1, 1)
            ctx_mix = (z_cand * pi_w).sum(dim=1)
            #pi_l2 = pi_hat.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt().view(B, 1, 1)
            #ctx_mix = ctx_mix / pi_l2
            ctx_mix = _ste_hard_threshold(ctx_mix, float(self.ctx_mix_eps))
            ctx = z_t + ctx_mix
            pi = pi_hat
            #print(self.router_tau, self.router_k)
            #print(t_start,taus_cand, pi_hat.argmax(dim=-1)[0], pi_hat[pi_hat.argmax(dim=-1)])
            # π-weighted sum of per-state embeddings, so the denoiser sees a
            # smooth signal indicating *which* history positions were picked.
            # ``taus_cand`` has shape [K] with absolute diffusion indices; the
            # lookup yields [K, H] and we reduce to [B, H] via π_hat.
            #print(pi_hat)
            #print(pi_hat.shape,s_cand.shape,s_cand[pi_hat.argmax(dim=-1)], pi_hat.to(dtype=torch.float32) @ s_cand , pi_hat.argmax(dim=-1),print(pi_hat))
            #state_cond = s_cur + pi_hat.to(dtype=torch.float32) @ s_cand            # [B, H]
            state_cond = None
        seq_in = ctx / ctx.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        if t_cond is None:
            t_b = torch.full((B,), int(t_start), device=device, dtype=torch.long)
        elif isinstance(t_cond, float):
            t_b = torch.full((B,), float(t_cond), device=device, dtype=torch.float32)
        else:
            t_b = torch.full((B,), int(t_cond), device=device, dtype=torch.long)
        if torch.is_floating_point(t_b):
            t_cont = t_b.to(dtype=torch.float32).clamp(min=0.0, max=1.0)
        else:
            t_cont = timestep_index_to_float(t_b, self.num_timesteps)
        cls_inp = None
        if self.num_labels is not None and self.num_labels > 0 and labels is not None:
            cls_inp = labels

        logits = self.cnn(seq_in, t_cont, cls_inp, state_cond=state_cond)

        # ``seq_in`` is the simplex actually fed into the CNN (current view +
        # routed history, renormalised). Return it so the sampler can use
        # ``seq_in > 0`` (instead of ``x_t > 0``) as the Bernoulli support
        # mask -- this lets routed history revive channels the current view
        # had zeroed out.
        return logits, pi, None, loss_bal, seq_in


class ActivityAuxHead(nn.Module):
    """Optional predictor on mean-pooled denoiser hidden state."""

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, num_classes))

    def forward(self, h_tokens: torch.Tensor) -> torch.Tensor:
        h = h_tokens.mean(dim=1)
        return self.net(h)
