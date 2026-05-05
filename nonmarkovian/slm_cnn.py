"""Vendored SLM ``CNNModel`` from ``SLM/models/dna_models.py`` (byte-aligned API).

``_feature_map_bhl`` exposes the pre-``final_conv`` map for FBD ``encoder`` only (not used in
training forward). ``tokens_to_four_channel_simplex`` and ``timestep_index_to_float`` adapt
discrete diffusion inputs to SLM's expected seq / t format.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from nonmarkovian.vocab import MASK_IDX


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 30.0) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)[...]


class CNNModel(nn.Module):
    """1D ResNet-style CNN — same structure as ``SLM/models/dna_models.py`` ``CNNModel``."""

    def __init__(self, alphabet_size, num_cls, num_cnn_stacks, classifier=False):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.classifier = classifier
        self.num_cls = num_cls

        self.clean_data = classifier
        self.cls_expanded_simplex = False
        self.hidden_dim = 128
        self.mode = "new_diff"
        self.dropout = 0.0
        self.cls_free_guidance = True
        self.num_cnn_stacks = num_cnn_stacks

        if self.clean_data:
            self.linear = nn.Embedding(self.alphabet_size, embedding_dim=self.hidden_dim)
        else:
            expanded_simplex_input = self.cls_expanded_simplex or not classifier and (
                self.mode == "dirichlet" or self.mode == "riemannian"
            )
            inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1)
            if (self.mode == "ardm" or self.mode == "lrar") and not classifier:
                inp_size += 1
            self.linear = nn.Conv1d(inp_size, self.hidden_dim, kernel_size=9, padding=4)
            self.time_embedder = nn.Sequential(
                GaussianFourierProjection(embed_dim=self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        self.num_layers = 5 * self.num_cnn_stacks
        self.convs = [
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=64, padding=256),
        ]
        self.convs = nn.ModuleList(
            [copy.deepcopy(layer) for layer in self.convs for _ in range(self.num_cnn_stacks)]
        )
        self.time_layers = nn.ModuleList(
            [Dense(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.final_conv = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(
                self.hidden_dim,
                self.hidden_dim if classifier else self.alphabet_size,
                kernel_size=1,
            ),
        )
        self.dropout = nn.Dropout(self.dropout)
        if classifier:
            self.cls_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.num_cls),
            )

        if self.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(
                num_embeddings=self.num_cls + 1, embedding_dim=self.hidden_dim
            )
            self.cls_layers = nn.ModuleList(
                [Dense(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)]
            )

    def _feature_map_bhl(
        self,
        seq: torch.Tensor,
        t: torch.Tensor,
        cls,
        state_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Feature map [B, H, L] after residual stack, before ``final_conv`` (generative or classifier).

        ``state_cond`` is an optional ``[B, hidden_dim]`` additive tensor that is
        folded into the per-layer time-conditioning pathway. Callers use this to
        inject **diffusion-state positional** information (e.g. a π-weighted
        sum of learnable per-state embeddings indicating *which* history
        positions the router picked). Ignored for ``clean_data`` (classifier).
        """
        if self.clean_data:
            feat = self.linear(seq)
            feat = feat.permute(0, 2, 1)
            time_emb = None
            cls_emb = None
        else:
            if t.dim() > 1:
                t = t.squeeze(-1)
            time_emb = F.relu(self.time_embedder(t))
            if state_cond is not None:
                if state_cond.shape[0] != time_emb.shape[0] or state_cond.shape[-1] != time_emb.shape[-1]:
                    raise ValueError(
                        f"state_cond shape {tuple(state_cond.shape)} incompatible with "
                        f"time_emb {tuple(time_emb.shape)}"
                    )
                
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))
            cls_emb = None
            if self.cls_free_guidance and not self.classifier:
                if cls is None:
                    cls = seq.new_full((seq.shape[0],), self.num_cls, dtype=torch.long)
                cls_emb = self.cls_embedder(cls)

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            if self.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]
            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h
        return feat

    def forward(self, seq, t, cls=None, return_embedding=False, state_cond=None):
        feat = self._feature_map_bhl(seq, t, cls, state_cond=state_cond)
        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)
        if self.classifier:
            feat = feat.mean(dim=1)
            if return_embedding:
                embedding = self.cls_head[:1](feat)
                return self.cls_head[1:](embedding), embedding
            else:
                return self.cls_head(feat)
        return feat


def tokens_to_four_channel_simplex(x: torch.Tensor) -> torch.Tensor:
    """Map token ids [B, L] (ACGT + MASK) to a 4-channel input for the CNN.

    Real bases use one-hot; MASK uses a uniform 0.25 simplex (absorbing state).
    """
    B, L = x.shape
    device, dtype = x.device, torch.float32
    base = x.clamp(max=3)
    oh = F.one_hot(base, num_classes=4).to(dtype=dtype, device=device)
    m = (x == MASK_IDX).to(dtype=dtype).unsqueeze(-1)
    unif = torch.full((B, L, 4), 0.25, device=device, dtype=dtype)
    return oh * (1.0 - m) + unif * m


def timestep_index_to_float(t_idx: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """Map integer diffusion indices to [0, 1] scalars per batch element."""
    if num_timesteps <= 1:
        return torch.zeros(t_idx.shape[0], device=t_idx.device, dtype=torch.float32)
    return (t_idx.float() + 0.5) / float(num_timesteps)
