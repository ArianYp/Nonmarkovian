"""Fréchet Biological Distance (FBD-style) and sequence embeddings for validation.

Dirichlet FM / SLM-style evaluation uses classifier hidden states as sequence embeddings and reports
Fréchet distance between real and generated distributions (analogous to FID). Use either the diffusion
model's ``encoder`` callable (default — runs DiT backbone with t=0 conditioning) or a frozen **FBCNN**
classifier from ``--fbcnn_ckpt`` (fly-brain 81-way: ``CNNModel(4, 81, 1, classifier=True)`` — one stack).
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
    """FBCNN sequence embeddings (Dirichlet-FM / SLM baseline style). x: [B, L] token ids 0..3.

    Matches the baseline ``CNNModel(..., classifier=True)`` eval path: run the **full** padded tensor
    through ``Embedding -> convs -> mean(dim=1) -> cls_head[:1]``. Padding uses token0 (A); those
    positions participate in the conv and global mean, same as ``collate_pad`` + batched validation.
    Rows that are entirely pad (rare) are zeroed.
    """
    device = x.device
    cnn.eval()
    B = x.shape[0]
    x = x.long().clamp(0, 3)
    t = torch.zeros(B, device=device, dtype=torch.float32)
    with torch.no_grad():
        _, emb = cnn(x, t, cls=None, return_embedding=True)
        if mask_pad.any():
            dead = mask_pad.all(dim=1)
            if dead.any():
                emb = emb.clone()
                emb[dead] = 0
    return emb


def frechet_distance_np(real: np.ndarray, gen: np.ndarray) -> float:
    """Fréchet distance between two sets of embeddings (FID-style). real: [n, d], gen: [m, d]."""
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    mu1 = real.mean(axis=0)
    mu2 = gen.mean(axis=0)
    sigma1 = np.cov(real, rowvar=False)
    sigma2 = np.cov(gen, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr = np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(ssdiff + tr)
