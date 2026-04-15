"""DiT (Diffusion Transformer) building blocks matching the MDLM / Dirichlet FM baseline.

Provides DDiTBlock (with AdaLN conditioning), Rotary positional embeddings,
TimestepEmbedder, LabelEmbedder, EmbeddingLayer, and DDitFinalLayer.
Falls back to PyTorch SDPA when flash_attn is not installed.
"""

from __future__ import annotations

import contextlib
import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import flash_attn.flash_attn_interface
    import flash_attn.layers.rotary

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# ---------------------------------------------------------------------------
# JIT helpers (match paper)
# ---------------------------------------------------------------------------

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Rotary positional embeddings
# ---------------------------------------------------------------------------


class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached: int | None = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)
        assert self.cos_cached is not None and self.sin_cached is not None
        return self.cos_cached, self.sin_cached


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary to a single tensor [B, L, H, D] using cos/sin [L, D]."""
    L = x.shape[1]
    cos = cos[:L].unsqueeze(0).unsqueeze(2)
    sin = sin[:L].unsqueeze(0).unsqueeze(2)
    return (x * cos) + (_rotate_half(x) * sin)


def _attention_with_rotary(
    qkv: torch.Tensor,
    rotary_cos_sin: tuple[torch.Tensor, torch.Tensor],
    batch_size: int,
    seq_len: int,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """Run attention with rotary embeddings, flash or SDPA fallback. qkv: [B, L, 3, H, D]."""
    cos, sin = rotary_cos_sin
    if HAS_FLASH_ATTN and qkv.is_cuda:
        cos_half = cos[0, :, 0, 0, : cos.shape[-1] // 2]
        sin_half = sin[0, :, 0, 0, : sin.shape[-1] // 2]
        ctx = torch.cuda.amp.autocast(enabled=False)
        with ctx:
            qkv = flash_attn.layers.rotary.apply_rotary_emb_qkv_(
                qkv, cos_half.to(qkv.dtype), sin_half.to(qkv.dtype)
            )
        qkv_flat = rearrange(qkv, "b s ... -> (b s) ...")
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=qkv.device
        )
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv_flat, cu_seqlens, seq_len, dropout_p if training else 0.0, causal=False
        )
        return rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)
    else:
        cos_full = cos[0, :, 0, 0, :]
        sin_full = sin[0, :, 0, 0, :]
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = _apply_rotary_single(q, cos_full, sin_full)
        k = _apply_rotary_single(k, cos_full, sin_full)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p if training else 0.0
        )
        x = x.transpose(1, 2).contiguous()
        return rearrange(x, "b s h d -> b s (h d)")


# ---------------------------------------------------------------------------
# Layer primitives
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class EmbeddingLayer(nn.Module):
    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding[x]


# ---------------------------------------------------------------------------
# Conditioning embedders
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Class-label embedder with +1 slot for unconditional (classifier-free guidance)."""

    def __init__(self, num_classes: int, cond_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


# ---------------------------------------------------------------------------
# DiT blocks
# ---------------------------------------------------------------------------


class DDiTBlock(nn.Module):
    """Transformer block with AdaLN conditioning (matches paper exactly)."""

    def __init__(self, dim: int, n_heads: int, cond_dim: int, dim_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        if dim_ff is None:
            dim_ff = 4 * dim

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_ff, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim_ff, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_p = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return bias_dropout_add_scale_fused_train if self.training else bias_dropout_add_scale_fused_inference

    def forward(self, x: torch.Tensor, rotary_cos_sin, c: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)[
            :, None
        ].chunk(6, dim=2)

        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
        x = _attention_with_rotary(qkv, rotary_cos_sin, batch_size, seq_len, self.dropout_p, self.training)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout_p)

        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout_p,
        )
        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Autocast helper
# ---------------------------------------------------------------------------


def amp_context(device: torch.device):
    """bfloat16 autocast on CUDA, no-op elsewhere."""
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return contextlib.nullcontext()
