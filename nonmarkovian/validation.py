"""Held-out validation: diffusion loss and optional FBD on generated vs real embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from nonmarkovian.forward import sample_all_views
from nonmarkovian.metrics import encoder_mean_pool_embeddings, fbcnn_embed_sequences, frechet_distance_np

if TYPE_CHECKING:
    from argparse import Namespace

    from nonmarkovian.fbcnn import CNNModel
    from nonmarkovian.model import RoutedDenoiser
    from nonmarkovian.simple_model import DiscreteDenoiser


def train_val_split(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Subset, Subset]:
    """Random train/val split. val_fraction in (0, 1)."""
    n = len(dataset)
    if n < 2:
        raise ValueError("Dataset must have at least 2 samples for train/val split")
    n_val = max(1, int(round(n * val_fraction)))
    if n_val >= n:
        n_val = max(1, n // 10)
    n_train = n - n_val
    if n_train < 1:
        raise ValueError("val_fraction too large for this dataset size")
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=g)


def timestep_loss_weight(alphas: torch.Tensor, t_start: int) -> float:
    if t_start == 0:
        return float(alphas[0].item())
    return float((alphas[t_start] - alphas[t_start - 1]).clamp(min=1e-6).item())


@torch.no_grad()
def validate_routed(
    model: "RoutedDenoiser",
    val_loader: DataLoader,
    alphas: torch.Tensor,
    device: torch.device,
    aux_head,
    args: "Namespace",
    *,
    epoch: int,
    global_step: int,
) -> dict[str, float]:
    """Average diffusion (and optional aux) loss on val_loader."""
    model.eval()
    if aux_head is not None:
        aux_head.eval()
    total = 0.0
    total_diff = 0.0
    total_aux = 0.0
    n_aux_batches = 0
    n_batches = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(global_step + epoch * 10007 + 12345)

    for batch in val_loader:
        x0 = batch["x0"].to(device)
        pad = batch["mask_pad"].to(device)
        labels = batch.get("label")
        if labels is not None:
            labels = labels.to(device)

        views = sample_all_views(x0, alphas, generator=gen)
        t_start = int(torch.randint(0, args.num_timesteps, (1,), device=device).item())
        logits, _pi, h_dec, _loss_bal = model(views, t_start, labels=labels)

        target = x0.clamp(max=3)
        ce = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
        ce = ce.masked_fill(pad, 0.0)
        denom = (~pad).float().sum().clamp(min=1.0)
        w = timestep_loss_weight(alphas, t_start)
        diff_loss = ce.float().sum() / denom * w
        loss = diff_loss

        if aux_head is not None and labels is not None and args.aux_beta > 0:
            aux_logits = aux_head(h_dec)
            aux_loss = F.cross_entropy(aux_logits, labels)
            loss = loss + args.aux_beta * aux_loss
            total_aux += float(aux_loss.item())
            n_aux_batches += 1

        total += float(loss.item())
        total_diff += float(diff_loss.item())
        n_batches += 1

    out: dict[str, float] = {
        "val/loss": total / max(n_batches, 1),
        "val/diff_loss": total_diff / max(n_batches, 1),
    }
    if n_aux_batches:
        out["val/aux_loss"] = total_aux / n_aux_batches
    return out


@torch.no_grad()
def validate_simple(
    model: "DiscreteDenoiser",
    val_loader: DataLoader,
    alphas: torch.Tensor,
    device: torch.device,
    aux_head,
    args: "Namespace",
    *,
    epoch: int,
    global_step: int,
) -> dict[str, float]:
    from nonmarkovian.forward import corrupt_sequence

    model.eval()
    if aux_head is not None:
        aux_head.eval()
    total = 0.0
    total_diff = 0.0
    total_aux = 0.0
    n_aux_batches = 0
    n_batches = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(global_step + epoch * 10007 + 12345)

    for batch in val_loader:
        x0 = batch["x0"].to(device)
        pad = batch["mask_pad"].to(device)
        labels = batch.get("label")
        if labels is not None:
            labels = labels.to(device)

        t_start = int(torch.randint(0, args.num_timesteps, (1,), device=device).item())
        x_t = corrupt_sequence(x0, float(alphas[t_start].item()), generator=gen)
        t_b = torch.full((x0.shape[0],), t_start, device=device, dtype=torch.long)
        logits, h_dec = model(x_t, t_b, labels=labels)

        target = x0.clamp(max=3)
        ce = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
        ce = ce.masked_fill(pad, 0.0)
        denom = (~pad).float().sum().clamp(min=1.0)
        w = timestep_loss_weight(alphas, t_start)
        diff_loss = ce.float().sum() / denom * w
        loss = diff_loss

        if aux_head is not None and labels is not None and args.aux_beta > 0:
            aux_logits = aux_head(h_dec)
            aux_loss = F.cross_entropy(aux_logits, labels)
            loss = loss + args.aux_beta * aux_loss
            total_aux += float(aux_loss.item())
            n_aux_batches += 1

        total += float(loss.item())
        total_diff += float(diff_loss.item())
        n_batches += 1

    out: dict[str, float] = {
        "val/loss": total / max(n_batches, 1),
        "val/diff_loss": total_diff / max(n_batches, 1),
    }
    if n_aux_batches:
        out["val/aux_loss"] = total_aux / n_aux_batches
    return out


@torch.no_grad()
def compute_fbd_routed(
    model: "RoutedDenoiser",
    val_loader: DataLoader,
    alphas: torch.Tensor,
    device: torch.device,
    args: "Namespace",
    *,
    n_samples: int,
    seq_len: int,
    epoch: int,
    fbcnn: "CNNModel | None" = None,
) -> float:
    from nonmarkovian.sample import sample_sequences

    model.eval()
    encoder = model.encoder
    real_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    collected = 0
    for batch in val_loader:
        if collected >= n_samples:
            break
        x0 = batch["x0"].to(device)
        pad = batch["mask_pad"].to(device)
        labels = batch.get("label")
        b = x0.shape[0]
        take = min(b, n_samples - collected)
        if fbcnn is not None:
            real_chunks.append(fbcnn_embed_sequences(fbcnn, x0[:take], pad[:take]))
        else:
            real_chunks.append(encoder_mean_pool_embeddings(encoder, x0[:take], pad[:take]))
        if labels is not None:
            label_chunks.append(labels[:take].to(device))
        collected += take
    if collected < 2:
        return float("nan")

    real_emb = torch.cat(real_chunks, dim=0)[:n_samples]

    labels_tensor: torch.Tensor | None = None
    if label_chunks:
        labels_tensor = torch.cat(label_chunks, dim=0)[:n_samples]

    gen_parts: list[torch.Tensor] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 424242 + epoch * 100003)
    for start in range(0, n_samples, args.val_gen_batch):
        bs = min(args.val_gen_batch, n_samples - start)
        lab = labels_tensor[start : start + bs] if labels_tensor is not None else None
        g = sample_sequences(
            model,
            alphas,
            bs,
            seq_len,
            device,
            labels=lab,
            generator=gen,
        )
        pad_g = torch.zeros(g.shape[0], g.shape[1], dtype=torch.bool, device=device)
        if fbcnn is not None:
            gen_parts.append(fbcnn_embed_sequences(fbcnn, g, pad_g))
        else:
            gen_parts.append(encoder_mean_pool_embeddings(encoder, g, pad_g))
    gen_emb = torch.cat(gen_parts, dim=0)

    r = real_emb.cpu().numpy().astype(np.float64)
    g_np = gen_emb.cpu().numpy().astype(np.float64)
    return frechet_distance_np(r, g_np)


@torch.no_grad()
def compute_fbd_simple(
    model: "DiscreteDenoiser",
    val_loader: DataLoader,
    alphas: torch.Tensor,
    device: torch.device,
    args: "Namespace",
    *,
    n_samples: int,
    seq_len: int,
    epoch: int,
    fbcnn: "CNNModel | None" = None,
) -> float:
    from nonmarkovian.sample_simple import sample_sequences_simple

    model.eval()
    encoder = model.encoder
    real_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    collected = 0
    for batch in val_loader:
        if collected >= n_samples:
            break
        x0 = batch["x0"].to(device)
        pad = batch["mask_pad"].to(device)
        labels = batch.get("label")
        b = x0.shape[0]
        take = min(b, n_samples - collected)
        if fbcnn is not None:
            real_chunks.append(fbcnn_embed_sequences(fbcnn, x0[:take], pad[:take]))
        else:
            real_chunks.append(encoder_mean_pool_embeddings(encoder, x0[:take], pad[:take]))
        if labels is not None:
            label_chunks.append(labels[:take].to(device))
        collected += take
    if collected < 2:
        return float("nan")

    real_emb = torch.cat(real_chunks, dim=0)[:n_samples]

    labels_tensor: torch.Tensor | None = None
    if label_chunks:
        labels_tensor = torch.cat(label_chunks, dim=0)[:n_samples]

    gen_parts: list[torch.Tensor] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 424242 + epoch * 100003)
    for start in range(0, n_samples, args.val_gen_batch):
        bs = min(args.val_gen_batch, n_samples - start)
        lab = labels_tensor[start : start + bs] if labels_tensor is not None else None
        g = sample_sequences_simple(
            model,
            alphas,
            bs,
            seq_len,
            device,
            labels=lab,
            generator=gen,
        )
        pad_g = torch.zeros(g.shape[0], g.shape[1], dtype=torch.bool, device=device)
        if fbcnn is not None:
            gen_parts.append(fbcnn_embed_sequences(fbcnn, g, pad_g))
        else:
            gen_parts.append(encoder_mean_pool_embeddings(encoder, g, pad_g))
    gen_emb = torch.cat(gen_parts, dim=0)

    r = real_emb.cpu().numpy().astype(np.float64)
    g_np = gen_emb.cpu().numpy().astype(np.float64)
    return frechet_distance_np(r, g_np)
