"""Fréchet Biological Distance (FBD-style) and sequence embeddings for validation.

Dirichlet FM / SLM-style evaluation uses classifier hidden states as sequence embeddings and reports
Fréchet distance between real and generated distributions (analogous to FID). Use either the diffusion
model's ``encoder`` callable (default — runs DiT backbone with t=0 conditioning) or a frozen **FBCNN**
classifier from ``--fbcnn_ckpt`` (fly-brain 81-way: ``CNNModel(4, 81, 4, classifier=True)``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg

if TYPE_CHECKING:
    from nonmarkovian.fbcnn import CNNModel


def encoder_mean_pool_embeddings(encoder: nn.Module, x: torch.Tensor, mask_pad: torch.Tensor) -> torch.Tensor:
    """Encode tokens and mean-pool over non-pad positions. x: [B, L], mask_pad True = pad -> [B, d]."""
    z = encoder(x)
    mask = (~mask_pad).float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (z * mask).sum(dim=1) / denom


def fbcnn_embed_sequences(cnn: "CNNModel", x: torch.Tensor, mask_pad: torch.Tensor) -> torch.Tensor:
    """Per-sequence embeddings from frozen CNN classifier; valid tokens only (ACGT indices 0..3). x: [B, L]."""
    device = x.device
    cnn.eval()
    B = x.shape[0]
    t = torch.zeros(B, device=device, dtype=torch.float32)
    x = x.long().clamp(0, 3)
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(B):
            n_valid = int((~mask_pad[i]).sum().item())
            if n_valid < 1:
                out.append(torch.zeros(1, cnn.hidden_dim, device=device, dtype=torch.float32))
                continue
            seq = x[i : i + 1, :n_valid]
            _, emb = cnn(seq, t[i : i + 1], cls=None, return_embedding=True)
            out.append(emb)
    return torch.cat(out, dim=0)


def frechet_distance_np(real: np.ndarray, gen: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance between two sets of embeddings (FID-style). real: [n, d], gen: [m, d]."""
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    mu1 = real.mean(axis=0)
    mu2 = gen.mean(axis=0)
    sigma1 = np.cov(real, rowvar=False)
    sigma2 = np.cov(gen, rowvar=False)
    d = sigma1.shape[0]
    sigma1 = sigma1 + np.eye(d, dtype=np.float64) * eps
    sigma2 = sigma2 + np.eye(d, dtype=np.float64) * eps
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr = np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(ssdiff + tr)
