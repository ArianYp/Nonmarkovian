"""Promoter generation with Non-Markovian routed diffusion (SLM new_diff).

Ports the SLM promoter experiment into the Nonmarkovian framework using the
same Boltzmann-router + Bernoulli-simplex logic as the enhancer pipeline
(train.py / model.py / forward.py), but with:

  - PromoterModel backbone  : dilated CNN conditioned on time + CAGE signal
  - RoutedDenoiserPromoter  : same router as RoutedDenoiserCNN + PromoterModel
  - PromoterDatasetWrapper  : wraps SLM's PromoterDataset → dict format
  - SEI H3K4me3 validation  : sp-mse between real and generated sequences

Data prerequisites (run once before training):
  1. Data is pre-prepared at /lustre/scratch126/cellgen/lotfollahi/ha11/dirichlet-flow-matching/data/promoter_design/
  2. Build genome memory-map:
       cd data_promoter/promoter_design && python make_genome_memmap.py

Dependencies: selene_sdk, pyBigWig, pytabix  (available in SLM virtualenv)

Run example:
  python -m nonmarkovian.nonmark_promoter \\
      --data_dir data_promoter/promoter_design \\
      --epochs 200 --batch_size 64 --T 1000 --sampling_steps 100
"""

from __future__ import annotations

import argparse
import atexit
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# ── SLM utilities (PromoterDataset, Sei) ───────────────────────────────────
# SLM repo lives as a sibling of the Nonmarkovian repo.
_SLM_ROOT = Path(__file__).resolve().parent.parent.parent / "SLM"
if _SLM_ROOT.is_dir() and str(_SLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLM_ROOT))

try:
    from promoter_utils.promoter_dataset import PromoterDataset as _SLMPromoterDataset
    from promoter_utils.sei import Sei
    from selene_sdk.utils import NonStrandSpecific

    _HAS_PROMOTER_DEPS = True
    _PROMOTER_IMPORT_ERR = ""
except ImportError as _err:
    _HAS_PROMOTER_DEPS = False
    _PROMOTER_IMPORT_ERR = str(_err)

# ── Nonmarkovian internals ──────────────────────────────────────────────────
from nonmarkovian.device_utils import cuda_is_usable, resolve_device_arg
from nonmarkovian.distributed_utils import (
    barrier,
    cleanup_process_group,
    setup_process_group,
    unwrap_ddp,
)
from nonmarkovian.forward import corrupt_sequence_bernoulli, sample_all_views_bernoulli
from nonmarkovian.slm_cnn import timestep_index_to_float, tokens_to_four_channel_simplex
from nonmarkovian.train_timing import tic, toc_ms

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


# ── tiny helper (avoids depending on SLM's esm.py) ─────────────────────────
def _strip_prefix(state_dict: dict, prefixes: tuple = ("module.",)) -> dict:
    pat = re.compile("^(" + "|".join(re.escape(p) for p in prefixes) + ")")
    return {pat.sub("", k): v for k, v in state_dict.items()}


# ══════════════════════════════════════════════════════════════════════════════
# PromoterModel
# Inline copy from SLM/models/promoter_model.py.  Only deps: torch + math.
# Architecture: 20-block (4 stacks × 5) dilated ResNet CNN with Gaussian
# Fourier time embedding and CAGE signal conditioning.
# ══════════════════════════════════════════════════════════════════════════════

class _GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 30.0) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B]  →  [B, embed_dim]
        x_proj = x[:, None] * self.W[None, :] * 2.0 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class _Dense(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


class PromoterModel(nn.Module):
    """Dilated-ResNet CNN denoiser for promoter generation.

    Matches SLM's ``PromoterModel`` exactly (4 stacks × 5 dilated blocks,
    256-dim hidden, GaussianFourier time embedding).

    Args:
        embed_dim:       Time-embedding size (default 256).
        n_hidden:        CNN feature channels (default 256).
        alphabet_size:   Nucleotide vocabulary (default 4).
        signal_channels: CAGE signal channels concatenated to seq input (default 1).
    """

    # Dilation pattern repeated across 4 stacks (5 blocks per stack = 20 total)
    _DILATIONS = [1, 1, 4, 16, 64]
    _PADDINGS  = [4, 4, 16, 64, 256]

    def __init__(
        self,
        embed_dim: int = 256,
        n_hidden: int = 256,
        alphabet_size: int = 4,
        signal_channels: int = 1,
    ) -> None:
        super().__init__()
        self.alphabet_size = alphabet_size
        n = n_hidden

        self.embed = nn.Sequential(
            _GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        inp_size = alphabet_size + signal_channels
        self.linear = nn.Conv1d(inp_size, n, kernel_size=9, padding=4)

        dils = self._DILATIONS * 4    # 4 stacks
        pads = self._PADDINGS  * 4
        self.blocks = nn.ModuleList(
            [nn.Conv1d(n, n, kernel_size=9, dilation=d, padding=p) for d, p in zip(dils, pads)]
        )
        self.denses = nn.ModuleList([_Dense(embed_dim, n) for _ in range(len(self.blocks))])
        self.norms  = nn.ModuleList([nn.GroupNorm(1, n)   for _ in range(len(self.blocks))])

        self.act = nn.SiLU()
        self.final = nn.Sequential(
            nn.Conv1d(n, n, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(n, alphabet_size, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        signal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:      [B, L, alphabet_size]  simplex probabilities (routed context).
            t:      [B]                    normalized time in (0, 1].
            signal: [B, L, signal_channels] CAGE signal.
        Returns:
            logits: [B, L, alphabet_size]  raw (un-normalised) logits.
        """
        embed = self.act(self.embed(t / 2.0))           # [B, embed_dim]
        inp   = torch.cat([x, signal], dim=-1)           # [B, L, alpha+sig]
        out   = self.act(self.linear(inp.permute(0, 2, 1)))  # [B, n, L]

        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            h = self.act(block(norm(out + dense(embed)[:, :, None])))
            out = h + out if h.shape == out.shape else h

        out = self.final(out).permute(0, 2, 1)           # [B, L, alphabet_size]
        out = out - out.mean(dim=-1, keepdim=True)        # centre logits (SLM convention)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# RoutedDenoiserPromoter
# Same Boltzmann router as RoutedDenoiserCNN (model.py); uses PromoterModel
# as the inner denoiser and passes the CAGE signal through to it.
# ══════════════════════════════════════════════════════════════════════════════

def _ste_hard_threshold(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Straight-through zero-out of sub-threshold entries (same as model.py)."""
    if eps <= 0.0:
        return x
    mask = (x.abs() > eps).to(dtype=x.dtype)
    return x - (x * (1.0 - mask)).detach()


class RoutedDenoiserPromoter(nn.Module):
    """Boltzmann router + PromoterModel denoiser.

    Architecture mirrors ``RoutedDenoiserCNN`` exactly.  The only difference
    is that the inner denoiser is ``PromoterModel`` (which takes an extra
    ``signal`` argument for CAGE conditioning) instead of ``CNNModel``.

    Router: W_cur / W_phi Conv1d projections → dot-product compatibility →
    Gumbel-softmax (train) / argmax (eval) → π-weighted context mix → denoiser.
    """

    def __init__(
        self,
        *,
        max_len: int = 1024,
        num_timesteps: int = 1000,
        embed_dim: int = 256,
        n_hidden: int = 256,
        router_tau: float = 1.0,
        router_k: int = 1,
        router_conv_kernel: int = 1,
        router_out_channels: int = 128,
        signal_channels: int = 1,
    ) -> None:
        super().__init__()
        self.max_len        = max_len
        self.num_timesteps  = num_timesteps
        self.router_tau     = router_tau
        self.router_k       = int(router_k)
        self.ctx_mix_eps: float = 1e-4

        rk = router_conv_kernel
        if rk < 1 or rk % 2 == 0:
            raise ValueError("router_conv_kernel must be a positive odd integer (e.g. 1, 3, 5, 9)")
        pad = rk // 2
        self.router_conv_kernel  = rk
        self.router_out_channels = router_out_channels

        self.W_cur = nn.Conv1d(4, router_out_channels, kernel_size=rk, padding=pad, bias=False)
        self.W_phi = nn.Conv1d(4, router_out_channels, kernel_size=rk, padding=pad, bias=False)

        self.promoter_model = PromoterModel(
            embed_dim=embed_dim,
            n_hidden=n_hidden,
            alphabet_size=4,
            signal_channels=signal_channels,
        )

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_simplex(x: torch.Tensor) -> torch.Tensor:
        """Cast [B, L] token ids or [B, L, 4] simplex to float [B, L, 4]."""
        if x.ndim == 3:
            return x.to(dtype=torch.float32)
        return tokens_to_four_channel_simplex(x)

    def _embed_current_and_candidates(
        self,
        x_views: torch.Tensor,
        t_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Extract z_t [B,L,4] and z_cand [B,K,L,4] from the views buffer."""
        if x_views.ndim == 4:
            B, T, L, _ = x_views.shape
        else:
            B, T, L = x_views.shape
        device = x_views.device

        z_t = self._to_simplex(x_views[:, t_start])  # [B, L, 4]
        K   = T - t_start - 1
        if K <= 0:
            return z_t, None, None

        cand = x_views[:, t_start + 1 : T]
        z_cand = cand.to(dtype=torch.float32) if cand.ndim == 4 else None
        if z_cand is None:
            raise ValueError("x_views must have 4 channels (simplex)")
        taus_cand = torch.arange(t_start + 1, T, device=device, dtype=torch.long)
        return z_t, z_cand, taus_cand

    def _compatibility_scores(
        self,
        z_t:    torch.Tensor,   # [B, L, 4]
        z_cand: torch.Tensor,   # [B, K, L, 4]
    ) -> torch.Tensor:
        """Per-batch compatibility e [B, K] via W_cur / W_phi dot similarity."""
        B, Kc, L, _ = z_cand.shape
        c_out    = self.router_out_channels
        inv_sqrt = 1.0 / math.sqrt(float(L * c_out))

        if int(self.router_conv_kernel) == 1:
            # Pointwise (kernel=1): collapse to a 4×4 bilinear form (memory-efficient)
            wc = self.W_cur.weight.squeeze(-1)   # [C_out, 4]
            wp = self.W_phi.weight.squeeze(-1)   # [C_out, 4]
            M  = wc.transpose(0, 1) @ wp         # [4, 4]
            zt_proj = torch.einsum("bli,ij->blj", z_t, M)
            return torch.einsum("blj,bklj->bk", zt_proj, z_cand) * inv_sqrt

        h_cur  = self.W_cur(z_t.transpose(1, 2).contiguous())               # [B, C, L]
        z_ck   = z_cand.reshape(B * Kc, L, 4).transpose(1, 2).contiguous()
        h_cand = self.W_phi(z_ck).view(B, Kc, c_out, L)
        return (h_cur[:, None, :, :] * h_cand).sum(dim=(2, 3)) * inv_sqrt   # [B, K]

    def _router_forward(
        self, e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gumbel-softmax (train) / one-hot argmax (eval) routing."""
        tau     = max(self.router_tau, 1e-6)
        pi_soft = torch.softmax(e / tau, dim=-1)
        if self.training:
            pi = F.gumbel_softmax(e, tau=tau, dim=-1, hard=False)
        else:
            pi = F.one_hot(e.argmax(dim=-1), num_classes=e.shape[-1]).to(dtype=e.dtype)
        return pi, pi_soft

    def _load_balance_loss(
        self, e: torch.Tensor, pi_soft: torch.Tensor
    ) -> torch.Tensor:
        """Switch-style load-balancing loss (same as model.py)."""
        B, K = e.shape
        if K == 0:
            return e.new_tensor(0.0)
        k_hard = e.argmax(dim=-1)
        f      = F.one_hot(k_hard, num_classes=K).to(dtype=e.dtype).mean(dim=0)
        bar_pi = pi_soft.mean(dim=0)
        return (float(K) * (f * bar_pi).sum()).to(e.dtype)

    def _scheduler_corruption(
        self, t_cond, num_classes: int = 4, scheduler: str = "loglinear"
    ) -> torch.Tensor:
        """Corruption fraction at ``t_cond`` (0 = clean/final, 1 = fully noised/start).

        Matches ``RoutedDenoiserCNN._scheduler_corruption`` / the Bernoulli
        expected-collision curve in ``_expected_nums``.
        """
        t = t_cond if torch.is_tensor(t_cond) else torch.tensor(float(t_cond))
        if scheduler == "loglinear":
            expect_nums = torch.exp(torch.log(torch.tensor(float(num_classes), device=t.device)) * t)
        else:  # "linear"
            expect_nums = float(num_classes) * t
        expect_nums = torch.clamp(expect_nums, min=1.0)
        corruption = (expect_nums - 1.0) / float(max(num_classes - 1, 1))
        return torch.clamp(corruption, 0.0, 1.0)

    # ── forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        x_views:     torch.Tensor,
        t_start:     int,
        signal:      torch.Tensor,
        t_cond:      float | None = None,
        t_start_abs: int   | None = None,
        scheduler:   str          = "loglinear",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_views:     [B, T′, L, 4] Bernoulli simplex views (T′ = T - t_start_abs).
            t_start:     Index into x_views for the "current" view (usually 0).
            signal:      [B, L, signal_channels] CAGE signal conditioning.
            t_cond:      Float time in (0,1]; None → derive from t_start / T.
            t_start_abs: Absolute timestep index when x_views is a window.

        Returns:
            logits:   [B, L, 4]
            pi:       [B, K] routing weights (K=0 when no candidates)
            loss_bal: scalar load-balancing loss
            seq_in:   [B, L, 4] renormalised context fed to the denoiser
        """
        if x_views.ndim == 4:
            B, T, L, _ = x_views.shape
        else:
            B, T, L = x_views.shape
        device    = x_views.device
        t_abs     = int(t_start) if t_start_abs is None else int(t_start_abs)

        # ── time (shared by the corruption blend and denoiser conditioning) ─
        if t_cond is None:
            t_float = float(t_abs + 1) / float(self.num_timesteps)
        else:
            t_float = float(t_cond)

        z_t, z_cand, _ = self._embed_current_and_candidates(x_views, t_start)

        # ── routing ───────────────────────────────────────────────────────
        if z_cand is None:
            ctx      = z_t
            pi       = z_t.new_zeros(B, 0)
            loss_bal = z_t.new_tensor(0.0)
        else:
            e             = self._compatibility_scores(z_t, z_cand)
            pi_hat, pi_s  = self._router_forward(e)
            loss_bal      = self._load_balance_loss(e, pi_s) if self.training else e.new_tensor(0.0)
            ctx_mix       = (z_cand * pi_hat.view(B, -1, 1, 1)).sum(dim=1)
            ctx_mix       = _ste_hard_threshold(ctx_mix, self.ctx_mix_eps)

            # Corruption-scheduled blend of current view + routed history, gated to
            # non-clean positions only -- identical to RoutedDenoiserCNN. A clean
            # position is a one-hot (max == 1); Bernoulli-corrupted / masked positions
            # are multi-hot with max < 1, so history is mixed in only where it helps.
            corruption    = self._scheduler_corruption(t_float, num_classes=4, scheduler=scheduler)
            w_cur         = 1.0 + corruption
            w_hist        = 1.0 - corruption
            is_masked     = (z_t.max(-1).values < 1.0)
            ctx           = torch.where(is_masked[..., None], w_hist * ctx_mix + w_cur * z_t, z_t)
            pi            = pi_hat

        seq_in = ctx / ctx.sum(dim=-1, keepdim=True).clamp(min=1e-8)  
        t_b = torch.full((B,), t_float, device=device, dtype=torch.float32)

        logits = self.promoter_model(seq_in, t_b, signal)   # [B, L, 4]

        # Zero-weighted touch on router weights so DDP sees them participate
        # in the autograd graph even when K = 0 (no candidates this step).
        # Without this, t_start = num_timesteps - 1 → router weights skipped →
        # DDP error: "Expected to have finished reduction in the prior iteration".
        if self.training and z_cand is None:
            logits = logits + 0.0 * (self.W_cur.weight.sum() + self.W_phi.weight.sum())

        return logits, pi, loss_bal, seq_in


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class PromoterDatasetWrapper(Dataset):
    """Wraps SLM's ``PromoterDataset`` into the Nonmarkovian dict format.

    Each item:
        {"x0": LongTensor[L], "signal": FloatTensor[L, 1]}

    SLM's dataset returns np.float32 [L, 6]:
        cols 0:4  = one-hot nucleotides (A/C/G/T)
        cols 4:6  = CAGE plus / minus strand signals
    We use only the plus-strand signal (col 4) to match ``PromoterModel``'s
    single-channel conditioning (``inp_size = 4 + 1``).
    """

    def __init__(
        self,
        data_dir:   str | Path,
        split:      str = "train",
        seq_length: int = 1024,
        n_tsses:    int = 100_000,
    ) -> None:
        if not _HAS_PROMOTER_DEPS:
            raise ImportError(
                f"PromoterDataset dependencies not available ({_PROMOTER_IMPORT_ERR}). "
                "Install selene_sdk, pyBigWig, pytabix and make sure "
                f"the SLM directory is accessible at {_SLM_ROOT}."
            )
        self._inner = _SLMPromoterDataset(
            data_dir=str(data_dir),
            seqlength=seq_length,
            split=split,
            n_tsses=n_tsses,
        )
        self._seq_length = seq_length

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict:
        raw   = self._inner[idx]                       # np.float32 [L, 6]
        x0    = torch.from_numpy(raw[:, :4]).argmax(dim=-1).long()   # [L]
        sig   = torch.from_numpy(raw[:, 4:5])          # [L, 1]  (plus strand)
        return {"x0": x0, "signal": sig}

    @property
    def seq_length(self) -> int:
        return self._seq_length


def collate_promoter(batch: list[dict]) -> dict:
    return {
        "x0":    torch.stack([b["x0"]    for b in batch]),   # [B, L]
        "signal": torch.stack([b["signal"] for b in batch]), # [B, L, 1]
    }


# ══════════════════════════════════════════════════════════════════════════════
# Reverse sampling
# Mirrors sample.py / sample_simple.py logic adapted for promoter signal.
# ══════════════════════════════════════════════════════════════════════════════

def _sample_bernoulli(
    probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Identical to sample.py._sample_bernoulli — returns a bool tensor."""
    if generator is None:
        u = torch.rand_like(probs)
    else:
        u = torch.rand(probs.shape, device=probs.device, dtype=probs.dtype, generator=generator)
    return u < probs


def _expected_nums(
    t: torch.Tensor,
    num_classes: int = 4,
    scheduler:   str = "loglinear",
) -> torch.Tensor:
    """Identical to sample.py._expected_nums."""
    if scheduler == "loglinear":
        return torch.clamp(
            torch.exp(torch.log(torch.tensor(float(num_classes), device=t.device)) * t),
            min=1.0,
        )
    if scheduler == "linear":
        return torch.clamp(float(num_classes) * t, min=1.0)
    raise ValueError(f"Unknown scheduler: {scheduler!r}")


@torch.no_grad()
def sample_promoter(
    model:               RoutedDenoiserPromoter,
    signal:              torch.Tensor,
    num_steps:           int,
    num_timesteps_train: int,
    device:              torch.device,
    seq_len:             int,
    vocab_size:          int = 4,
    scheduler:           str = "loglinear",
    history_mode:        str = "trajectory",
    corruption_mode:     str = "independent",
    independent_threshold: float = 0.6,
    generator:           torch.Generator | None = None,
) -> torch.Tensor:
    """SLM new_diff reverse sampling for the routed promoter model.

    Mirrors ``sample_sequences`` from ``sample.py`` exactly, with the only
    change being that ``signal`` is threaded through every model call for
    CAGE conditioning (the promoter task has no labels / CFG).

    Loop structure (same as sample.py):
      - ``i`` steps forward from 1 → T
      - ``t_start = T - i``  (T-1 down to 0)
      - ``t_val   = 1 - (i-1)/T``  (1.0 down to 1/T)
      - support mask for logits:    ``(x_t > 0)``
      - support mask for Bernoulli: ``(x_t > 0)`` (trajectory), or dropped after
        ``independent_threshold`` of the steps when ``corruption_mode='independent'``
      - final denoising is a separate model call after the loop

    Args:
        signal:  [B, L, 1] CAGE signal (on ``device``).
        num_steps:           Reverse-process steps.
        num_timesteps_train: T used during training (= views buffer size).
        history_mode:        ``'trajectory'`` or ``'uniform'``.
        corruption_mode:     ``'independent'`` or ``'trajectory'`` (mirrors
            ``sample.py``). In ``'independent'`` mode the ``(x_t > 0)`` support
            mask on the Bernoulli draw is dropped once ``i`` passes
            ``independent_threshold * num_steps`` (the noisy region), letting
            positions re-activate channels — matches the enhancer sampler.
        independent_threshold: fraction of reverse steps after which the
            ``independent`` support mask is dropped (default 0.6; sample.py's
            legacy ``threshold=6`` with ``6 * num_steps // 10``).
        generator:           Optional torch.Generator for reproducibility.

    Returns:
        ids: [B, L] LongTensor of predicted nucleotide tokens.
    """
    model.eval()
    B     = signal.shape[0]
    T     = int(num_steps)
    C     = int(vocab_size)

    # Align the model's internal timestep count to num_steps (mirrors sample.py line 329)
    old_num_timesteps = getattr(model, "num_timesteps", None)
    model.num_timesteps = T

    # ── initialise ────────────────────────────────────────────────────────
    x_t         = torch.full((B, seq_len, C), 1.0 / C, device=device, dtype=torch.float32)
    hat_x0_ids  = torch.zeros((B, seq_len),           device=device, dtype=torch.long)
    views_buffer = x_t.new_full((B, T, seq_len, C),   1.0 / C)   # [B, T, L, C]

    # ── main reverse loop  ─────────────────────────────────────────────────
    for i in range(1, T + 1):
        t_val   = 1.0 - float(i - 1) / float(T)   # 1.0 → 1/T  (same as sample.py)
        t_start = T - i                            # T-1 → 0

        # write current x_t into the views slot (trajectory history mode);
        # 'uniform' keeps non-current slots at 1/C (matches sample.py).
        if history_mode == "uniform":
            views_buffer.fill_(1.0 / float(C))
        views_buffer[:, t_start] = x_t

        # router + denoiser forward — full buffer, real t_start index
        logits, _pi, _lb, seq_in = model(
            views_buffer, t_start, signal=signal, t_cond=t_val, scheduler=scheduler,
        )

        # ── logit masking: (x_t > 0)  — identical to sample.py ──────────
        support_mask = (x_t > 0)
        has_any      = support_mask.any(dim=-1, keepdim=True)
        support_mask = torch.where(has_any, support_mask, torch.ones_like(support_mask))
        logits = logits.masked_fill(~support_mask, torch.finfo(logits.dtype).min)

        model_prob = F.softmax(logits, dim=-1)
        # renormalise if numerical drift
        prob_sum = model_prob.sum(dim=-1, keepdim=True)
        if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-4):
            model_prob = model_prob / prob_sum.clamp(min=1e-8)

        hat_x0_ids = model_prob.argmax(dim=-1).clamp(max=C - 1)   # [B, L]

        # ── SLM reverse step (carry + Bernoulli sample) ───────────────────
        t3        = torch.full((B, 1, 1), t_val, device=device, dtype=torch.float32)
        nominator   = _expected_nums(t3 - 1.0 / T, C, scheduler) - 1.0
        denominator = torch.clamp(_expected_nums(t3, C, scheduler) - 1.0, min=1e-8)
        weight    = torch.clamp(nominator / denominator, min=0.0, max=1.0)
        predicted = torch.clamp(model_prob + weight * (1.0 - model_prob), min=0.0, max=1.0)

        # Bernoulli sample with (x_t > 0) support mask — identical to sample.py.
        # 'independent' mode drops the mask in the noisy region so positions can
        # re-activate channels; 'trajectory' always keeps the mask.
        support_bern = (x_t > 0)
        if corruption_mode == "independent" and i > independent_threshold * T:
            sample_pred = _sample_bernoulli(predicted, generator)
        else:
            sample_pred = _sample_bernoulli(predicted, generator) & support_bern
        sample_pred_sum = sample_pred.sum(dim=-1, keepdim=True)             # [B, L, 1]
        fallback      = F.one_hot(predicted.argmax(dim=-1), num_classes=C).to(dtype=torch.bool)
        sample_pred   = torch.where(sample_pred_sum > 0, sample_pred, fallback)
        x_t           = sample_pred.to(dtype=torch.float32)
        x_t           = x_t / x_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # ── final denoising step (separate, same as sample.py) ────────────────
    t_last = 1.0 / float(T)
    if history_mode == "uniform":
        views_buffer.fill_(1.0 / float(C))
    views_buffer[:, 0] = x_t
    logits_last, _pi, _lb, _seq_in = model(
        views_buffer, 0, signal=signal, t_cond=t_last, scheduler=scheduler,
    )
    ids = logits_last.argmax(dim=-1).clamp(max=C - 1)

    # restore original timestep count so training is unaffected
    if old_num_timesteps is not None:
        model.num_timesteps = old_num_timesteps

    return ids


# ══════════════════════════════════════════════════════════════════════════════
# SEI validation (H3K4me3 profile MSE = sp-mse)
# ══════════════════════════════════════════════════════════════════════════════

def _load_sei(sei_path: str | Path, device: torch.device):
    """Load the Sei (4096-bp, 21907-feature) chromatin-profile model."""
    sei   = NonStrandSpecific(Sei(4096, 21907))
    ckpt  = torch.load(str(sei_path), map_location="cpu")
    sd    = _strip_prefix(ckpt["state_dict"], prefixes=("module.",))
    sei.load_state_dict(sd)
    sei   = sei.to(device).eval()
    return sei


@torch.no_grad()
def _get_sei_profile(
    sei,
    h3k4me3_mask: np.ndarray,
    seq_one_hot:  torch.Tensor,
    device:       torch.device,
) -> np.ndarray:
    """Mean H3K4me3 score per sequence via the Sei model.

    Args:
        seq_one_hot: [B, L, 4] float tensor.
        h3k4me3_mask: bool [21907] selecting H3K4me3 features.
    Returns:
        [B] numpy float32 array of mean H3K4me3 scores.
    """
    B, L, _ = seq_one_hot.shape
    pad     = 1536  # Sei requires 4096-bp input; centre 1024 bp + 1536 flanks
    seq_t   = seq_one_hot.to(device).transpose(1, 2)               # [B, 4, L]
    context = torch.ones(B, 4, pad, device=device) * 0.25
    inp     = torch.cat([context, seq_t, context], dim=2)          # [B, 4, 4096]
    out     = sei(inp).cpu().float().numpy()                       # [B, 21907]
    return out[:, h3k4me3_mask].mean(axis=1)                       # [B]


@torch.no_grad()
def validate_promoter(
    model:           RoutedDenoiserPromoter,
    val_loader:      DataLoader,
    device:          torch.device,
    args:            argparse.Namespace,
    *,
    epoch:           int = 0,
    global_step:     int = 0,
    sei              = None,
    h3k4me3_mask:    np.ndarray | None = None,
    max_sei_batches: int = 4,
) -> dict:
    """Compute val/loss (new_diff NLL) and optionally val/sp_mse (SEI MSE).

    Val loss mirrors the training forward pass (same Bernoulli corruption at
    a random t, same cross-entropy × T loss) evaluated under ``no_grad``.

    Returns a dict of metric names → float values.
    """
    model.eval()
    T           = int(args.T)
    scheduler   = args.bernoulli_scheduler
    without_T   = args.without_T
    num_ts      = args.num_timesteps

    total_nll    = 0.0
    total_tokens = 0
    sp_mse_vals: list[float] = []

    for batch_idx, batch in enumerate(val_loader):
        x0     = batch["x0"].to(device)       # [B, L]
        signal = batch["signal"].to(device)   # [B, L, 1]
        B, L   = x0.shape

        # ── validation NLL  ──────────────────────────────────────────────
        gen = torch.Generator(device=device).manual_seed(epoch * 10_000 + batch_idx)
        t_start = int(torch.randint(0, num_ts, (1,), generator=gen, device=device).item())
        t_cont  = float(t_start + 1) / float(num_ts)

        views = sample_all_views_bernoulli(
            x0, num_ts, t_start=t_start,
            scheduler=scheduler, generator=gen,
            corruption_mode=args.corruption_mode,
        )                                       # [B, num_ts - t_start, L, 4]

        logits, _, _, _ = model(
            views, 0, signal=signal, t_cond=t_cont, t_start_abs=t_start, scheduler=scheduler
        )
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, x0[:, :, None]).squeeze(-1)  # [B, L]
        if not without_T:
            nll = float(T) * nll
        total_nll    += float(nll.sum().item())
        total_tokens += B * L

        # ── SEI sp-mse  ──────────────────────────────────────────────────
        # max_sei_batches < 0 ⇒ run on all batches (full validation set).
        sei_active = (sei is not None and h3k4me3_mask is not None
                      and (max_sei_batches < 0 or batch_idx < max_sei_batches))
        if sei_active:
            real_oh   = F.one_hot(x0, num_classes=4).float()
            real_sc   = _get_sei_profile(sei, h3k4me3_mask, real_oh, device)

            gen_ids   = sample_promoter(
                model, signal,
                num_steps=args.sampling_steps,
                num_timesteps_train=num_ts,
                device=device, seq_len=L,
                scheduler=scheduler,
                history_mode=args.history_mode,
                corruption_mode=args.corruption_mode,
                independent_threshold=args.independent_threshold,
            )
            gen_oh    = F.one_hot(gen_ids, num_classes=4).float()
            gen_sc    = _get_sei_profile(sei, h3k4me3_mask, gen_oh, device)
            sp_mse_vals.append(float(((real_sc - gen_sc) ** 2).mean()))

    val_loss = total_nll / max(total_tokens, 1)
    metrics: dict = {"val/loss": val_loss, "epoch": epoch + 1}
    if sp_mse_vals:
        metrics["val/sp_mse"] = float(np.mean(sp_mse_vals))
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# EMA  (identical to train.py._EMA)
# ══════════════════════════════════════════════════════════════════════════════

class _EMA:
    def __init__(self, params, decay: float) -> None:
        self.params  = [p for p in params if p.requires_grad]
        self.decay   = float(decay)
        self.shadow  = [p.detach().clone() for p in self.params]
        self.backup: list | None = None

    @torch.no_grad()
    def update(self) -> None:
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self) -> None:
        self.backup = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def copy_to(self) -> None:
        for p, s in zip(self.params, self.shadow):
            p.data.copy_(s)

    @torch.no_grad()
    def restore(self) -> None:
        if self.backup is None:
            return
        for p, b in zip(self.params, self.backup):
            p.data.copy_(b)
        self.backup = None


# ══════════════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train RoutedDenoiserPromoter on CAGE-conditioned promoter sequences.",
    )
    # Data
    p.add_argument("--data_dir",    type=str,
                   default="/lustre/scratch126/cellgen/lotfollahi/ha11/dirichlet-flow-matching/data/promoter_design",
                   help="Path to promoter_design/ (must contain genome mmap + SEI weights).")
    p.add_argument("--n_tsses",     type=int, default=100_000)
    p.add_argument("--seq_length",  type=int, default=1024)
    # Training
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--ema_decay",   type=float, default=0.9999,
                   help="EMA decay (0 = disable).")
    # Diffusion
    p.add_argument("--T",               type=int,   default=1000,
                   help="Diffusion timesteps T (scales the NLL loss).")
    p.add_argument("--num_timesteps",   type=int,   default=1000,
                   help="Views buffer size for routing (usually == T).")
    p.add_argument("--sampling_steps",  type=int,   default=100,
                   help="Reverse-process steps used during validation sampling.")
    p.add_argument("--bernoulli_scheduler", type=str, default="loglinear",
                   choices=("loglinear", "linear"))
    p.add_argument("--without_T",       action="store_true",
                   help="Do not scale NLL loss by T (matches SLM training.without_T).")
    p.add_argument("--corruption_mode", type=str, default="independent",
                   choices=("independent", "trajectory"),
                   help="Corruption strategy for both training views and reverse "
                        "sampling (matches train.py / enhancer). 'independent': "
                        "fresh i.i.d. noise per timestep; 'trajectory': shared "
                        "monotone noise draw.")
    p.add_argument("--independent_threshold", type=float, default=0.6,
                   help="Fraction of reverse steps after which the (x_t>0) support "
                        "mask is dropped in --corruption_mode independent (sample.py "
                        "legacy threshold=6 == 0.6).")
    p.add_argument("--history_mode",    type=str, default="trajectory",
                   choices=("trajectory", "uniform"),
                   help="History fill mode for reverse sampling views buffer.")
    # Router
    p.add_argument("--router_tau",          type=float, default=0.01)
    p.add_argument("--router_k",            type=int,   default=1,
                   help="Kept for CLI parity with train.py; routing uses full softmax mix.")
    p.add_argument("--router_conv_kernel",  type=int,   default=1,
                   help="W_cur / W_phi Conv1d kernel (odd; 1 = pointwise bilinear).")
    p.add_argument("--router_out_channels", type=int,   default=256)
    p.add_argument("--router_lambda_bal",   type=float, default=0.0,
                   help="Switch-style load-balancing loss weight (0 = off).")
    # Model
    p.add_argument("--embed_dim",   type=int, default=256,
                   help="Fourier time-embedding and hidden dim for PromoterModel.")
    p.add_argument("--n_hidden",    type=int, default=256,
                   help="CNN feature channels in PromoterModel.")
    # Validation / checkpointing
    p.add_argument("--val_batch_size",  type=int, default=0,
                   help="Validation batch size (0 = same as --batch_size).")
    p.add_argument("--val_epoch_freq",  type=int, default=1,
                   help="Run val/loss every N epochs.")
    p.add_argument("--sei_epoch_freq",  type=int, default=10,
                   help="Run SEI sp-mse every N epochs (expensive; 0 = disable).")
    p.add_argument("--max_sei_batches", type=int, default=4,
                   help="Max val batches used for SEI scoring (-1 = all).")
    p.add_argument("--best_mse_epochs", type=int, default=50,
                   help="In the last N epochs run SEI on ALL val batches every epoch "
                        "and save a best-sp_mse checkpoint.")
    p.add_argument("--save",    type=str, default="checkpoints/promoter.pt")
    p.add_argument("--seed",    type=int, default=0)
    # W&B
    p.add_argument("--wandb",     dest="use_wandb", action="store_true",  default=True)
    p.add_argument("--no-wandb",  dest="use_wandb", action="store_false")
    p.add_argument("--wandb_project",  type=str, default="nonmarkovian_promoter")
    p.add_argument("--wandb_run_name", type=str, default="")
    # Misc
    p.add_argument("--device",      type=str, default="auto")
    p.add_argument("--log_timing",  action="store_true")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_run_dir(requested: str | Path, use_wandb: bool) -> Path:
    """Per-run checkpoint directory so repeat/concurrent runs never overwrite each other.

    ``checkpoints/promoter.pt`` -> ``checkpoints/promoter_<runid>/`` and every
    checkpoint (final / best / best_mse) is written *inside* that directory with
    its clean basename. The run id is the active W&B run id when available (so the
    directory matches the W&B run); otherwise a generated id (``wandb.util.generate_id``
    or a uuid fallback).

    NOTE: resolve this ONCE per run and reuse the result — the generated-id
    fallback is non-deterministic, so calling it repeatedly would scatter
    checkpoints across several directories.
    """
    requested = Path(requested)
    run_id = None
    if use_wandb and wandb is not None and getattr(wandb, "run", None) is not None:
        run_id = wandb.run.id
    if not run_id:
        try:
            run_id = wandb.util.generate_id() if wandb is not None else None
        except Exception:
            run_id = None
    if not run_id:
        import uuid

        run_id = uuid.uuid4().hex[:8]
    return requested.parent / f"{requested.stem}_{run_id}"


def _to_float(x: torch.Tensor | float) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    return float(x)


def _train_loop(
    args:          argparse.Namespace,
    device:        torch.device,
    model:         RoutedDenoiserPromoter,
    train_loader:  DataLoader,
    val_loader:    DataLoader | None,
    use_wandb:     bool,
    *,
    rank:          int = 0,
    ddp:           bool = False,
    train_sampler: DistributedSampler | None = None,
) -> None:

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = _EMA(unwrap_ddp(model).parameters(), decay=float(args.ema_decay)) \
          if float(args.ema_decay) > 0 else None

    # ── load Sei model for sp-mse (rank-0 only) ───────────────────────────
    sei           = None
    h3k4me3_mask  = None
    if rank == 0:
        sei_path    = Path(args.data_dir) / "best.sei.model.pth.tar"
        names_path  = Path(args.data_dir) / "target.sei.names"
        sei_enabled = (
            args.sei_epoch_freq > 0
            and sei_path.is_file()
            and names_path.is_file()
            and _HAS_PROMOTER_DEPS
        )
        if sei_enabled:
            try:
                import pandas as pd
                feats        = pd.read_csv(str(names_path), sep="|", header=None)
                h3k4me3_mask = (feats[1].str.strip().values == "H3K4me3")
                sei          = _load_sei(sei_path, device)
                print(f"  SEI loaded from {sei_path}  "
                      f"(H3K4me3 features: {h3k4me3_mask.sum()})")
            except Exception as exc:
                print(f"  Warning: could not load SEI model ({exc}). sp-mse disabled.")
        elif args.sei_epoch_freq > 0 and rank == 0:
            if not _HAS_PROMOTER_DEPS:
                print(f"  Warning: SEI disabled (promoter deps missing: {_PROMOTER_IMPORT_ERR}).")
            elif not sei_path.is_file():
                print(f"  Warning: SEI model not found at {sei_path}. sp-mse disabled.")

    best_val_loss   = float("inf")
    best_save_path: Path | None = None
    best_mse        = float("inf")
    best_mse_path: Path | None = None
    global_step     = 0

    # Resolve the per-run checkpoint directory ONCE (rank 0), so best/best_mse/final
    # all land together and repeat runs never overwrite each other.
    base_name = final_path = best_path = best_mse_path_target = None
    if rank == 0:
        run_dir   = _resolve_run_dir(args.save, use_wandb)
        run_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(args.save).name                       # e.g. promoter.pt
        stem, suffix = Path(base_name).stem, Path(base_name).suffix
        final_path          = run_dir / base_name
        best_path           = run_dir / f"{stem}.best{suffix}"
        best_mse_path_target = run_dir / f"{stem}.best_mse{suffix}"
        print(f"checkpoints -> {run_dir}/  "
              f"(final={base_name}, best={stem}.best{suffix}, best_mse={stem}.best_mse{suffix})")
        if use_wandb:
            wandb.summary["checkpoint_dir"] = str(run_dir.resolve())

    T         = int(args.T)
    scheduler = args.bernoulli_scheduler

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_batches  = 0
        num_batches = len(train_loader)
        sum_ms_views = sum_ms_fwd = sum_ms_loss = sum_ms_bwd = 0.0

        for batch_idx, batch in enumerate(train_loader):
            x0     = batch["x0"].to(device)       # [B, L]
            signal = batch["signal"].to(device)   # [B, L, 1]
            B, L   = x0.shape

            gen = torch.Generator(device=device)
            gen.manual_seed(global_step + epoch * 100_000)

            t_start = int(torch.randint(0, args.num_timesteps, (1,), device=device).item())
            t_cont  = float(t_start + 1) / float(args.num_timesteps)

            # ── corruption / views ────────────────────────────────────────
            if args.log_timing:
                t0 = tic(device)
            views = sample_all_views_bernoulli(
                x0, args.num_timesteps, t_start=t_start,
                scheduler=scheduler, generator=gen,
                corruption_mode=args.corruption_mode,
            )                                           # [B, T - t_start, L, 4]
            ms_views = toc_ms(t0, device) if args.log_timing else 0.0

            # ── model forward ─────────────────────────────────────────────
            if args.log_timing:
                t0 = tic(device)
            logits, pi, loss_bal, seq_in = model(
                views, 0, signal=signal, t_cond=t_cont, t_start_abs=t_start, scheduler=scheduler
            )
            ms_fwd = toc_ms(t0, device) if args.log_timing else 0.0

            # ── new_diff cross-entropy loss ───────────────────────────────
            if args.log_timing:
                t0 = tic(device)
            log_probs = F.log_softmax(logits, dim=-1)
            nll       = -log_probs.gather(-1, x0[:, :, None]).squeeze(-1)   # [B, L]
            if not args.without_T:
                nll = float(T) * nll
            diff_loss = nll.float().sum() / float(B * L)

            loss = diff_loss
            if args.router_lambda_bal > 0 and loss_bal.numel() > 0:
                loss = loss + args.router_lambda_bal * loss_bal
            ms_loss = toc_ms(t0, device) if args.log_timing else 0.0

            # ── backward + step ───────────────────────────────────────────
            if args.log_timing:
                t0 = tic(device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update()
            ms_bwd = toc_ms(t0, device) if args.log_timing else 0.0

            if args.log_timing:
                sum_ms_views += ms_views
                sum_ms_fwd   += ms_fwd
                sum_ms_loss  += ms_loss
                sum_ms_bwd   += ms_bwd

            total_loss  += float(loss.item())
            n_batches   += 1
            global_step += 1

            if use_wandb:
                with torch.no_grad():
                    if pi.shape[-1] > 0:
                        p   = pi.clamp(min=1e-8)
                        ent = -(p * p.log()).sum(dim=-1).mean()
                    else:
                        ent = torch.tensor(0.0, device=device)
                log_payload: dict = {
                    "train/loss":           float(loss.item()),
                    "train/diff_loss":      float(diff_loss.item()),
                    "train/t_start":        t_start,
                    "train/lr":             opt.param_groups[0]["lr"],
                    "train/grad_norm":      _to_float(grad_norm),
                    "train/router_entropy": float(ent.item()),
                    "train/batch_idx":      batch_idx,
                    "train/batches_per_epoch": num_batches,
                    "epoch":                epoch + 1,
                }
                if args.router_lambda_bal > 0:
                    log_payload["train/loss_balance"] = float(loss_bal.item())
                if args.log_timing:
                    log_payload.update({
                        "train/time_ms_views":    ms_views,
                        "train/time_ms_forward":  ms_fwd,
                        "train/time_ms_loss":     ms_loss,
                        "train/time_ms_backward": ms_bwd,
                    })
                wandb.log(log_payload, step=global_step)

        avg = total_loss / max(n_batches, 1)
        if rank == 0:
            print(f"epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}")
        if args.log_timing and n_batches > 0 and rank == 0:
            print(
                f"  timing_ms (batch avg): views={sum_ms_views/n_batches:.1f}  "
                f"fwd={sum_ms_fwd/n_batches:.1f}  "
                f"loss={sum_ms_loss/n_batches:.1f}  "
                f"bwd={sum_ms_bwd/n_batches:.1f}"
            )
        if use_wandb:
            wandb.log({"train/epoch_loss_avg": avg, "epoch": epoch + 1}, step=global_step)

        # ── validation ────────────────────────────────────────────────────
        if val_loader is not None and (epoch + 1) % args.val_epoch_freq == 0:
            m  = unwrap_ddp(model)
            if ema is not None:
                ema.store()
                ema.copy_to()

            in_mse_window = (args.epochs - epoch) <= args.best_mse_epochs
            run_sei_periodic = (
                args.sei_epoch_freq > 0
                and (epoch + 1) % args.sei_epoch_freq == 0
            )
            run_sei = (
                sei is not None
                and h3k4me3_mask is not None
                and (in_mse_window or run_sei_periodic)
            )
            # In the best-mse window: SEI on the full val set every epoch.
            # Outside the window: keep the configured cap (default 4 batches).
            sei_batches_this_epoch = -1 if in_mse_window else args.max_sei_batches
            vmetrics = validate_promoter(
                m, val_loader, device, args,
                epoch=epoch, global_step=global_step,
                sei=sei if run_sei else None,
                h3k4me3_mask=h3k4me3_mask if run_sei else None,
                max_sei_batches=sei_batches_this_epoch,
            )

            if rank == 0:
                line = f"  val_loss={vmetrics['val/loss']:.4f}"
                if "val/sp_mse" in vmetrics:
                    line += f"  sp-mse={vmetrics['val/sp_mse']:.6f}"
                print(line)
                if use_wandb:
                    wandb.log(vmetrics, step=global_step)

                cur_val = float(vmetrics["val/loss"])
                if cur_val < best_val_loss:
                    best_val_loss = cur_val
                    best_save_path = best_path
                    best_save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model":           m.state_dict(),
                        "args":            vars(args),
                        "best_val_loss":   best_val_loss,
                        "best_epoch":      epoch + 1,
                        "best_global_step": global_step,
                    }, best_save_path)
                    print(f"  best checkpoint → {best_save_path}  (val/loss={best_val_loss:.4f})")
                    if use_wandb:
                        wandb.summary["checkpoint_best_path"]     = str(best_save_path.resolve())
                        wandb.summary["checkpoint_best_val_loss"] = best_val_loss

                # Best-MSE checkpoint — only inside the last-N-epochs window,
                # using the full val-set sp_mse (representative, not biased to
                # the first 4 batches like sei_epoch_freq evaluations).
                cur_mse = vmetrics.get("val/sp_mse")
                if in_mse_window and cur_mse is not None and float(cur_mse) < best_mse:
                    best_mse = float(cur_mse)
                    best_mse_path = best_mse_path_target
                    best_mse_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model":            m.state_dict(),
                        "args":             vars(args),
                        "best_mse":         best_mse,
                        "best_mse_epoch":   epoch + 1,
                        "best_mse_global_step": global_step,
                    }, best_mse_path)
                    print(f"  best-mse checkpoint → {best_mse_path}  (sp_mse={best_mse:.6f})")
                    if use_wandb:
                        wandb.summary["checkpoint_best_mse_path"] = str(best_mse_path.resolve())
                        wandb.summary["checkpoint_best_mse"]      = best_mse

            if ema is not None:
                ema.restore()

        if ddp:
            barrier()

    # ── save final checkpoint ─────────────────────────────────────────────
    if rank == 0:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": unwrap_ddp(model).state_dict(),
            "args":  vars(args),
        }, final_path)
        print(f"Saved final checkpoint: {final_path}")
        if best_save_path is not None:
            print(f"Best checkpoint:  {best_save_path}  (val/loss={best_val_loss:.4f})")
        if best_mse_path is not None:
            print(f"Best-mse checkpoint:  {best_mse_path}  (sp_mse={best_mse:.6f})")
    if ddp:
        barrier()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    ddp, rank, world_size, local_rank = setup_process_group()
    if ddp and not torch.cuda.is_available():
        cleanup_process_group()
        raise SystemExit("Multi-GPU training requires CUDA.")
    if ddp:
        atexit.register(cleanup_process_group)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = resolve_device_arg(args.device)

    torch.manual_seed(args.seed)

    use_wandb = bool(args.use_wandb and wandb is not None and rank == 0)
    if rank == 0 and args.use_wandb and wandb is None:
        print("wandb not installed; pip install wandb. Continuing without W&B.")

    # ── datasets ──────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    train_ds = PromoterDatasetWrapper(data_dir, split="train",
                                      seq_length=args.seq_length, n_tsses=args.n_tsses)
    val_ds   = PromoterDatasetWrapper(data_dir, split="valid",
                                      seq_length=args.seq_length, n_tsses=args.n_tsses)

    train_sampler: DistributedSampler | None = None
    if ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_promoter,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    vb = args.val_batch_size if args.val_batch_size > 0 else args.batch_size
    val_loader = DataLoader(
        val_ds, batch_size=vb, shuffle=False,
        collate_fn=collate_promoter, num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # ── model ─────────────────────────────────────────────────────────────
    model = RoutedDenoiserPromoter(
        max_len=args.seq_length,
        num_timesteps=args.num_timesteps,
        embed_dim=args.embed_dim,
        n_hidden=args.n_hidden,
        router_tau=args.router_tau,
        router_k=args.router_k,
        router_conv_kernel=args.router_conv_kernel,
        router_out_channels=args.router_out_channels,
        signal_channels=1,
    ).to(device)

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )

    n_params = sum(p.numel() for p in unwrap_ddp(model).parameters())
    if rank == 0:
        print(f"RoutedDenoiserPromoter: {n_params:,} parameters ({n_params / 1e6:.2f} M)")
        print(f"Dataset: {len(train_ds):,} train  {len(val_ds):,} val  "
              f"| seq_length={args.seq_length}  T={args.T}  "
              f"sampling_steps={args.sampling_steps}")
        if ddp:
            print(f"Distributed: world_size={world_size}  "
                  f"per-GPU batch={args.batch_size}  "
                  f"global_batch={args.batch_size * world_size}")

    # ── W&B ───────────────────────────────────────────────────────────────
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config=vars(args),
        )
        wandb.summary["model/total_parameters"] = n_params
        wandb.summary["device"] = str(device)
        wandb.summary["cuda_usable"] = cuda_is_usable()
        if device.type == "cuda":
            wandb.summary["cuda_device_name"] = torch.cuda.get_device_name(device)
        wandb.summary["distributed"] = ddp
        wandb.summary["world_size"]  = world_size

    try:
        _train_loop(
            args, device, model,
            train_loader, val_loader,
            use_wandb,
            rank=rank, ddp=ddp,
            train_sampler=train_sampler,
        )
    finally:
        if use_wandb:
            wandb.finish()

    if ddp:
        cleanup_process_group()


if __name__ == "__main__":
    main()
