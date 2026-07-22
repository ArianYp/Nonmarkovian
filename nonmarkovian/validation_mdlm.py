"""MDLM (masked / absorbing) validation + FBD, mirroring ``validation.py`` for the Bernoulli case.

Only the corruption-/sampler-dependent functions are reimplemented here:

* ``validate_simple_mdlm`` / ``validate_routed_mdlm`` — average **MDLM NELBO** on val data
  (cross-entropy on *masked* positions weighted by the MDLM time factor ``w(t)``).
* ``compute_fbd_simple_mdlm`` / ``compute_fbd_routed_mdlm`` — FBD on samples drawn with the MDLM
  ancestral samplers.
* ``print_epoch_diffusion_dna_samples_mdlm`` — DNA preview via the MDLM samplers.

DDP/embedding/Fréchet helpers (and the model-free preview / chance baselines) are imported
straight from ``validation.py`` so they are not duplicated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from nonmarkovian.forward_mdlm import (
    corrupt_mask,
    mdlm_alpha,
    mdlm_loss_weight,
    sample_all_views_mask,
)
from nonmarkovian.metrics import (
    encoder_mean_pool_embeddings,
    fbcnn_embed_sequences,
    frechet_distance_np,
)
from nonmarkovian.sample_mdlm import ids_to_strings, sample_sequences_mdlm
from nonmarkovian.sample_simple_mdlm import sample_sequences_simple_mdlm
from nonmarkovian.validation import (
    _all_reduce_sum_,
    _ddp_world_rank,
    _gather_concat_embeddings,
    _use_conditional_sampling_labels,
)
from nonmarkovian.vocab import MASK_IDX

if TYPE_CHECKING:
    from argparse import Namespace

    from nonmarkovian.fbcnn import CNNModel
    from nonmarkovian.model import RoutedDenoiserCNN
    from nonmarkovian.simple_model import DiscreteDenoiser, DiscreteDenoiserCNN


def _mdlm_masked_nll(
    logits: torch.Tensor,
    target: torch.Tensor,
    current_mask: torch.Tensor,
    t_val: float,
    *,
    scheduler: str,
    num_classes: int = 4,
) -> torch.Tensor:
    """Per-position MDLM NELBO contribution: ``w(t) * CE`` on masked positions, else 0.

    Returns ``[B, L]`` so callers can ``.sum() / (B*L)`` to match the Bernoulli per-token
    normalization used in ``validation.py``.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    ce = -torch.gather(log_probs, -1, target[:, :, None]).squeeze(-1)  # [B, L]
    w = mdlm_loss_weight(
        torch.tensor(float(t_val), device=logits.device),
        num_classes=num_classes,
        scheduler=scheduler,
    )
    return w * ce * current_mask.to(dtype=ce.dtype)


@torch.no_grad()
def validate_simple_mdlm(
    model: "DiscreteDenoiser | DiscreteDenoiserCNN",
    val_loader: DataLoader,
    device: torch.device,
    aux_head,
    args: "Namespace",
    *,
    epoch: int,
    global_step: int,
) -> dict[str, float]:
    model.eval()
    if aux_head is not None:
        aux_head.eval()
    dev_total = torch.zeros((), device=device, dtype=torch.float64)
    dev_total_diff = torch.zeros((), device=device, dtype=torch.float64)
    dev_total_aux = torch.zeros((), device=device, dtype=torch.float64)
    dev_n_aux = torch.zeros((), device=device, dtype=torch.float64)
    dev_n = torch.zeros((), device=device, dtype=torch.float64)

    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    gen = torch.Generator(device=device)
    gen.manual_seed(global_step + epoch * 10007 + 12345 + rank * 7919)
    scheduler = str(args.bernoulli_scheduler)
    nt = int(args.num_timesteps)

    for batch in val_loader:
        x0 = batch["x0"].to(device)
        if _use_conditional_sampling_labels(args):
            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(device)
        else:
            labels = None

        t_start = int(torch.randint(0, nt, (1,), device=device).item())
        t_val = float(t_start + 1) / float(nt)
        t_cont = torch.full((x0.shape[0], 1), t_val, device=device, dtype=torch.float32)
        x_t = corrupt_mask(x0, t_cont, scheduler=scheduler, generator=gen)
        current_mask = x_t == MASK_IDX

        labels_in = labels
        if labels is not None and getattr(args, "cond_drop_prob", 0.0) > 0:
            keep = torch.rand((x0.shape[0],), device=device) >= float(args.cond_drop_prob)
            if getattr(args, "backbone", "cnn") == "cnn":
                null_cls = int(getattr(args, "num_classes", 0))
                labels_in = torch.where(keep, labels, torch.full_like(labels, null_cls))
            elif not bool(keep.all()):
                labels_in = None

        logits, h_dec = model(x_t, t_cont.squeeze(-1), labels=labels_in)
        target = x0.clamp(max=3)
        denom = float(target.shape[0] * target.shape[1])
        nlog_p = _mdlm_masked_nll(logits, target, current_mask, t_val, scheduler=scheduler)
        diff_loss = nlog_p.float().sum() / denom
        loss = diff_loss

        if aux_head is not None and labels is not None and args.aux_beta > 0:
            aux_logits = aux_head(h_dec)
            aux_loss = F.cross_entropy(aux_logits, labels)
            loss = loss + args.aux_beta * aux_loss
            dev_total_aux += aux_loss.detach().double()
            dev_n_aux += 1.0

        dev_total += loss.detach().double()
        dev_total_diff += diff_loss.detach().double()
        dev_n += 1.0

    _all_reduce_sum_([dev_total, dev_total_diff, dev_total_aux, dev_n_aux, dev_n])
    n = max(float(dev_n.item()), 1.0)
    out: dict[str, float] = {
        "val/loss": float(dev_total.item()) / n,
        "val/diff_loss": float(dev_total_diff.item()) / n,
    }
    n_aux = float(dev_n_aux.item())
    if n_aux > 0:
        out["val/aux_loss"] = float(dev_total_aux.item()) / n_aux
    return out


@torch.no_grad()
def validate_routed_mdlm(
    model: "RoutedDenoiserCNN",
    val_loader: DataLoader,
    device: torch.device,
    aux_head,
    args: "Namespace",
    *,
    epoch: int,
    global_step: int,
) -> dict[str, float]:
    """MDLM NELBO on val data for the routed model, with both with-history and no-history flavors
    (mirrors ``validate_routed``). ``val/loss`` (with history) drives best-checkpoint selection."""
    model.eval()
    if aux_head is not None:
        aux_head.eval()
    dev_total = torch.zeros((), device=device, dtype=torch.float64)
    dev_total_diff = torch.zeros((), device=device, dtype=torch.float64)
    dev_total_diff_noh = torch.zeros((), device=device, dtype=torch.float64)
    dev_total_aux = torch.zeros((), device=device, dtype=torch.float64)
    dev_n_aux = torch.zeros((), device=device, dtype=torch.float64)
    dev_n = torch.zeros((), device=device, dtype=torch.float64)

    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    gen = torch.Generator(device=device)
    gen.manual_seed(global_step + epoch * 10007 + 12345 + rank * 7919)
    scheduler = str(args.bernoulli_scheduler)
    nt = int(args.num_timesteps)
    corruption_mode = str(getattr(args, "corruption_mode", "independent"))

    for batch in val_loader:
        x0 = batch["x0"].to(device)
        if _use_conditional_sampling_labels(args):
            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(device)
        else:
            labels = None

        t_start = int(torch.randint(0, nt, (1,), device=device).item())
        t_val = float(t_start + 1) / float(nt)

        views_full = sample_all_views_mask(
            x0, nt, scheduler=scheduler, generator=gen,
            corruption_mode=corruption_mode, return_simplex=True,
        )  # [B, nt, L, 4]
        B, Tv, L, _ = views_full.shape
        current_view = views_full[:, t_start].clone()
        views_noh = views_full.new_full((B, Tv, L, 4), 0.25)
        views_noh[:, t_start] = current_view

        labels_in = labels
        if labels is not None and getattr(args, "cond_drop_prob", 0.0) > 0:
            keep = torch.rand((B,), device=device) >= float(args.cond_drop_prob)
            if getattr(args, "backbone", "cnn") == "cnn":
                null_cls = int(getattr(args, "num_classes", 0))
                labels_in = torch.where(keep, labels, torch.full_like(labels, null_cls))
            elif not bool(keep.all()):
                labels_in = None

        target = x0.clamp(max=3)
        denom = float(target.shape[0] * target.shape[1])
        # Routed loss: alpha_{t-1}-weighted CE over ALL positions (no mask gating) — matches
        # train_mdlm.py (the non-Markovian model predicts every position via the denoiser).
        alpha_tm1 = mdlm_alpha(
            torch.tensor(t_val - 1.0 / float(nt), device=device), scheduler=scheduler
        )

        def _routed_nll(logits: torch.Tensor) -> torch.Tensor:
            ce = -torch.gather(F.log_softmax(logits, dim=-1), -1, target[:, :, None]).squeeze(-1)
            return (alpha_tm1 * ce).float().sum() / denom

        logits_full, _pi, h_dec, _lb, _seq = model(
            views_full, t_start, labels=labels_in, t_cond=t_val
        )
        diff_loss_full = _routed_nll(logits_full)

        logits_noh, _pi2, _h2, _lb2, _seq2 = model(
            views_noh, t_start, labels=labels_in, t_cond=t_val
        )
        diff_loss_noh = _routed_nll(logits_noh)

        loss = diff_loss_full
        if aux_head is not None and labels is not None and args.aux_beta > 0:
            aux_logits = aux_head(h_dec)
            aux_loss = F.cross_entropy(aux_logits, labels)
            loss = loss + args.aux_beta * aux_loss
            dev_total_aux += aux_loss.detach().double()
            dev_n_aux += 1.0

        dev_total += loss.detach().double()
        dev_total_diff += diff_loss_full.detach().double()
        dev_total_diff_noh += diff_loss_noh.detach().double()
        dev_n += 1.0

    _all_reduce_sum_(
        [dev_total, dev_total_diff, dev_total_diff_noh, dev_total_aux, dev_n_aux, dev_n]
    )
    n = max(float(dev_n.item()), 1.0)
    out: dict[str, float] = {
        "val/loss": float(dev_total.item()) / n,
        "val/diff_loss": float(dev_total_diff.item()) / n,
        "val/diff_loss_no_history": float(dev_total_diff_noh.item()) / n,
        "val/loss_no_history": float(dev_total_diff_noh.item()) / n,
    }
    n_aux = float(dev_n_aux.item())
    if n_aux > 0:
        out["val/aux_loss"] = float(dev_total_aux.item()) / n_aux
    return out


def _fbd_from_samples(
    real_chunks: list[torch.Tensor],
    gen_parts: list[torch.Tensor],
    device: torch.device,
    n_samples: int,
) -> float:
    """Gather + Fréchet distance shared by the MDLM FBD functions."""
    ddp_on, _ws, rk = _ddp_world_rank()
    real_emb_local = (
        torch.cat(real_chunks, dim=0) if real_chunks else torch.empty((0, 0), device=device)
    )
    real_emb = _gather_concat_embeddings(real_emb_local)
    if real_emb.shape[0] < 2:
        return float("nan")
    real_emb = real_emb[: int(n_samples)]

    emb_dim = real_emb.shape[1] if real_emb.ndim == 2 and real_emb.numel() else 0
    gen_emb_local = (
        torch.cat(gen_parts, dim=0) if gen_parts else torch.empty((0, emb_dim), device=device)
    )
    gen_emb = _gather_concat_embeddings(gen_emb_local)[: int(n_samples)]

    if ddp_on and rk != 0:
        return float("nan")
    if gen_emb.shape[0] < 2:
        return float("nan")
    r = real_emb.cpu().numpy().astype(np.float64)
    g_np = gen_emb.cpu().numpy().astype(np.float64)
    return frechet_distance_np(r, g_np)


@torch.no_grad()
def compute_fbd_routed_mdlm(
    model: "RoutedDenoiserCNN",
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
    """FBD for the routed MDLM model — symmetric to ``compute_fbd_routed`` but MDLM-sampled.
    ``alphas`` is used only for its length (= number of reverse steps)."""
    model.eval()
    encoder = model.encoder
    use_labs = _use_conditional_sampling_labels(args)
    _ddp_on, ws, rk = _ddp_world_rank()
    local_target = (int(n_samples) + ws - 1) // ws
    num_steps = int(alphas.shape[0])
    scheduler = str(getattr(args, "bernoulli_scheduler", "loglinear"))

    real_chunks: list[torch.Tensor] = []
    gen_parts: list[torch.Tensor] = []
    collected = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 424242 + int(epoch) * 100003 + rk * 101)
    for batch in val_loader:
        if collected >= local_target:
            break
        x0 = batch["x0"].to(device)
        pad = batch["mask_pad"].to(device)
        labels = batch.get("label")
        b = x0.shape[0]
        take = min(b, local_target - collected)
        if fbcnn is not None:
            real_chunks.append(fbcnn_embed_sequences(fbcnn, x0[:take], pad[:take]))
        else:
            real_chunks.append(encoder_mean_pool_embeddings(encoder, x0[:take], pad[:take]))
        lab = labels[:take].to(device) if (use_labs and labels is not None) else None
        g = sample_sequences_mdlm(
            model,
            num_steps,
            int(take),
            seq_len,
            device,
            num_timesteps_train=int(args.num_timesteps),
            labels=lab,
            guidance_scale=float(getattr(args, "guidance_scale", 0.0)),
            scheduler=scheduler,
            generator=gen,
            history_mode=str(getattr(args, "history_mode", "trajectory")),
            corruption_mode=str(getattr(args, "corruption_mode", "independent")),
            independent_threshold=float(getattr(args, "independent_threshold", 0.6)),
        )
        if fbcnn is not None:
            gen_parts.append(fbcnn_embed_sequences(fbcnn, g))
        else:
            gen_parts.append(encoder_mean_pool_embeddings(encoder, g))
        collected += take

    return _fbd_from_samples(real_chunks, gen_parts, device, n_samples)


@torch.no_grad()
def compute_fbd_simple_mdlm(
    model: "DiscreteDenoiser | DiscreteDenoiserCNN",
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
    """FBD for the simple MDLM model — symmetric to ``compute_fbd_simple`` but MDLM-sampled."""
    model.eval()
    encoder = model.encoder
    use_labs = _use_conditional_sampling_labels(args)
    _ddp_on, ws, rk = _ddp_world_rank()
    local_target = (int(n_samples) + ws - 1) // ws
    num_steps = int(alphas.shape[0])
    scheduler = str(getattr(args, "bernoulli_scheduler", "loglinear"))

    real_chunks: list[torch.Tensor] = []
    gen_parts: list[torch.Tensor] = []
    collected = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 424242 + int(epoch) * 100003 + rk * 101)
    for batch in val_loader:
        if collected >= local_target:
            break
        x0 = batch["x0"].to(device)
        pad = batch["mask_pad"].to(device)
        labels = batch.get("label")
        b = x0.shape[0]
        take = min(b, local_target - collected)
        if fbcnn is not None:
            real_chunks.append(fbcnn_embed_sequences(fbcnn, x0[:take], pad[:take]))
        else:
            real_chunks.append(encoder_mean_pool_embeddings(encoder, x0[:take], pad[:take]))
        lab = labels[:take].to(device) if (use_labs and labels is not None) else None
        g = sample_sequences_simple_mdlm(
            model,
            num_steps,
            int(take),
            seq_len,
            device,
            num_timesteps_train=int(args.num_timesteps),
            labels=lab,
            guidance_scale=float(getattr(args, "guidance_scale", 0.0)),
            scheduler=scheduler,
            generator=gen,
        )
        if fbcnn is not None:
            gen_parts.append(fbcnn_embed_sequences(fbcnn, g))
        else:
            gen_parts.append(encoder_mean_pool_embeddings(encoder, g))
        collected += take

    return _fbd_from_samples(real_chunks, gen_parts, device, n_samples)


@torch.no_grad()
def print_epoch_diffusion_dna_samples_mdlm(
    model: torch.nn.Module,
    alphas: torch.Tensor,
    device: torch.device,
    args: "Namespace",
    val_dataset: Dataset | None,
    *,
    epoch: int,
    global_step: int,
    n: int = 4,
    routed: bool,
    display_width: int = 96,
) -> None:
    """Decode ``n`` MDLM model samples after an epoch (routed vs simple sampler)."""

    def _trunc(s: str) -> str:
        if len(s) <= display_width:
            return s
        return f"{s[:display_width]}... ({len(s)} bp total)"

    labels: torch.Tensor | None = None
    if _use_conditional_sampling_labels(args):
        labs: list[int] = []
        if val_dataset is not None and len(val_dataset) > 0:
            for i in range(min(n, len(val_dataset))):
                item = val_dataset[i]
                labs.append(int(item["label"]) if "label" in item else 0)
        while len(labs) < n:
            labs.append(labs[-1] if labs else 0)
        labels = torch.tensor(labs[:n], device=device, dtype=torch.long)

    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 91000 + int(epoch) * 1009 + int(global_step))
    num_steps = int(alphas.shape[0])
    scheduler = str(getattr(args, "bernoulli_scheduler", "loglinear"))

    if routed:
        x = sample_sequences_mdlm(
            model, num_steps, n, args.max_len, device,
            num_timesteps_train=int(args.num_timesteps),
            labels=labels, scheduler=scheduler, generator=gen,
            history_mode=str(getattr(args, "history_mode", "trajectory")),
        )
    else:
        x = sample_sequences_simple_mdlm(
            model, num_steps, n, args.max_len, device,
            num_timesteps_train=int(args.num_timesteps),
            labels=labels, scheduler=scheduler, generator=gen,
        )

    strands = ids_to_strings(x)
    print(f"[samples-mdlm] epoch {epoch + 1} - model generations (n={n}, seq_len={args.max_len}):")
    for i, s in enumerate(strands):
        lab = ""
        if labels is not None:
            lab = f"  label={int(labels[i].item())}"
        print(f"  gen[{i}]{lab}")
        print(f"      {_trunc(s)}")
