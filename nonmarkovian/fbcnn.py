"""Fly-brain CNN classifier backbone for FBD-style embeddings (Dirichlet FM–style evaluation)."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B] or [B, 1]
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x_proj = x * self.W.unsqueeze(0) * 2.0 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


class CNNModel(nn.Module):
    """DNA CNN classifier; embedding = first block of cls_head (128-D) when return_embedding=True."""

    def __init__(self, alphabet_size: int, num_cls: int, num_cnn_stacks: int, classifier: bool = False):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.classifier = classifier
        self.num_cls = num_cls

        self.clean_data = classifier
        self.cls_expanded_simplex = False
        self.hidden_dim = 128
        self.mode = "new_diff"
        self.dropout_p = 0.0
        self.cls_free_guidance = True
        self.num_cnn_stacks = num_cnn_stacks

        if self.clean_data:
            self.linear = nn.Embedding(self.alphabet_size, embedding_dim=self.hidden_dim)
        else:
            expanded_simplex_input = self.cls_expanded_simplex or (
                not classifier and (self.mode == "dirichlet" or self.mode == "riemannian")
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
        conv_templates = [
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=64, padding=256),
        ]
        self.convs = nn.ModuleList(
            [copy.deepcopy(layer) for layer in conv_templates for _ in range(self.num_cnn_stacks)]
        )
        self.time_layers = nn.ModuleList([Dense(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])
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
        self.dropout = nn.Dropout(self.dropout_p)
        if classifier:
            self.cls_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.num_cls),
            )

        if self.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=self.hidden_dim)
            self.cls_layers = nn.ModuleList([Dense(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])

    def forward(
        self,
        seq: torch.Tensor,
        t: torch.Tensor,
        cls: torch.Tensor | None = None,
        return_embedding: bool = False,
    ):
        if self.clean_data:
            feat = self.linear(seq)
            feat = feat.permute(0, 2, 1)
        else:
            time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))

        if self.cls_free_guidance and not self.classifier:
            assert cls is not None
            cls_emb = self.cls_embedder(cls)

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            if self.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]
            h = self.norms[i](h.permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h
        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)
        if self.classifier:
            feat = feat.mean(dim=1)
            if return_embedding:
                embedding = self.cls_head[:1](feat)
                return self.cls_head[1:](embedding), embedding
            return self.cls_head(feat)
        return feat


def _extract_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dict or compatible mapping")
    for key in ("state_dict", "model_state_dict", "model"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]  # type: ignore[return-value]
    return ckpt  # type: ignore[return-value]


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
                break
        out[nk] = v
    return out


def load_fbcnn_classifier(
    ckpt_path: str | Path,
    device: torch.device,
    *,
    num_cls: int = 81,
    num_cnn_stacks: int = 4,
    alphabet_size: int = 4,
) -> CNNModel:
    """Load frozen CNNModel(alphabet_size, num_cls, num_cnn_stacks, classifier=True) from a PyTorch checkpoint."""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_sd = _extract_state_dict(ckpt)
    raw_sd = _strip_prefix(
        raw_sd,
        ("backbone.", "model.backbone.", "model.", "net.", "classifier.", "fbcnn."),
    )

    model = CNNModel(alphabet_size, num_cls, num_cnn_stacks, classifier=True).to(device)
    inc = model.load_state_dict(raw_sd, strict=False)
    missing = list(inc.missing_keys)
    unexpected = list(inc.unexpected_keys)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if missing or unexpected:
        import warnings

        warnings.warn(
            f"FBCNN load_state_dict incomplete: missing={len(missing)} unexpected={len(unexpected)} "
            f"(first missing: {missing[:5]}, first unexpected: {unexpected[:5]})",
            stacklevel=1,
        )
    return model
