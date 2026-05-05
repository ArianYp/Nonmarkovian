"""Held-out validation: diffusion loss and optional FBD on generated vs real embeddings."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from nonmarkovian.forward import corrupt_sequence_bernoulli, sample_all_views_bernoulli
from nonmarkovian.metrics import encoder_mean_pool_embeddings, fbcnn_embed_sequences, frechet_distance_np

if TYPE_CHECKING:
    from argparse import Namespace

    from nonmarkovian.fbcnn import CNNModel
    from nonmarkovian.model import RoutedDenoiser, RoutedDenoiserCNN
    from nonmarkovian.simple_model import DiscreteDenoiser, DiscreteDenoiserCNN
from nonmarkovian.sample import ids_to_strings

from nonmarkovian.sample import sample_sequences
from nonmarkovian.sample_simple import sample_sequences_simple


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


def _all_reduce_sum_(tensors: list[torch.Tensor]) -> None:
    """In-place SUM all-reduce across ranks when torch.distributed is initialized."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    for t in tensors:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


def _ddp_world_rank() -> tuple[bool, int, int]:
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_world_size(), dist.get_rank()
    return False, 1, 0


def _gather_concat_embeddings(emb_local: torch.Tensor) -> torch.Tensor:
    """All-gather variable-sized ``[N_local, D]`` embedding tensors across ranks and concatenate
    along dim 0 in rank order. Works with any backend (object-based gather over CPU)."""
    ddp_on, ws, _ = _ddp_world_rank()
    if not ddp_on:
        return emb_local
    x_cpu = emb_local.detach().cpu()
    gathered: list[torch.Tensor | None] = [None] * ws
    dist.all_gather_object(gathered, x_cpu)
    non_empty = [g for g in gathered if isinstance(g, torch.Tensor) and g.numel() > 0]
    if not non_empty:
        return emb_local
    out_cpu = torch.cat(non_empty, dim=0)
    return out_cpu.to(emb_local.device)


def _use_conditional_sampling_labels(args: "Namespace") -> bool:
    """Whether to pass activity labels into the denoiser during val sampling (FBD, DNA preview).

    ``--no_labels`` forces unconditional sampling even if ``num_classes`` were nonzero.
    """
    if getattr(args, "no_labels", False):
        return False
    return int(getattr(args, "num_classes", 0) or 0) > 0


def _expected_nums(t: torch.Tensor, *, num_classes: int = 4, scheduler: str = "loglinear") -> torch.Tensor:
    if scheduler == "loglinear":
        return torch.clamp(torch.exp(math.log(float(num_classes)) * t), min=1.0)
    if scheduler == "linear":
        return torch.clamp(float(num_classes) * t, min=1.0)
    raise ValueError(f"Unknown Bernoulli scheduler: {scheduler}")


def chance_validation_baselines(
    *,
    aux_beta: float = 0.0,
    num_classes: int = 0,
) -> dict[str, float]:
    """
    Expected diffusion loss if the model used uniform logits over the four nucleotides (mean CE =
    ``log(4)`` per token). ``val/chance_baseline_loss`` adds ``aux_beta * log(num_classes)`` when
    both are >0 (uniform auxiliary classifier), comparable to ``val/loss``.
    """
    diff = math.log(4.0)
    total = float(diff)
    if aux_beta > 0.0 and num_classes > 1:
        total = float(diff + aux_beta * math.log(float(num_classes)))
    return {
        "val/chance_baseline_diff": float(diff),
        "val/chance_baseline_loss": total,
    }


def print_val_and_random_dna_preview(
    val_dataset: Dataset,
    *,
    max_len: int,
    base_seed: int,
    n: int = 4,
    random_dna_seed_offset: int = 31337,
    display_width: int = 96,
) -> None:
    """Print the first ``n`` validation examples and ``n`` i.i.d. uniform random DNA strings (length ``max_len``)."""
    acgt = "ACGT"

    def _trunc(s: str) -> str:
        if len(s) <= display_width:
            return s
        return f"{s[:display_width]}... ({len(s)} bp total)"

    n_val = min(n, len(val_dataset))
    print(f"[preview] Validation split: showing {n_val} of {len(val_dataset)} sequences")
    for i in range(n_val):
        item = val_dataset[i]
        x = item["x0"]
        s = "".join(acgt[min(int(t), 3)] for t in x.flatten().tolist())
        lab = ""
        if "label" in item:
            lab = f"  label={int(item['label'])}"
        print(f"  val[{i}] len={len(s)}{lab}")
        print(f"      {_trunc(s)}")

    seed = int(base_seed) + int(random_dna_seed_offset)
    print(
        f"[preview] Uniform random DNA (same seed convention as FBD random baseline: base_seed+{random_dna_seed_offset}={seed}), length={max_len}"
    )
    g = torch.Generator()
    g.manual_seed(seed)
    for i in range(n):
        idx = torch.randint(0, 4, (max_len,), generator=g)
        s = "".join(acgt[int(t)] for t in idx.tolist())
        print(f"  random[{i}] len={len(s)}")
        print(f"      {_trunc(s)}")


@torch.no_grad()
def print_epoch_diffusion_dna_samples(
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
    """Decode ``n`` model samples after an epoch (routed vs simple sampler).

    ``alphas`` should be the **sampling** schedule (length ``num_timesteps_sample``); training
    timestep count comes from ``args.num_timesteps`` for denoiser conditioning.
    """

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

    nt_train = int(getattr(args, "num_timesteps", 0)) or int(alphas.shape[0])
    if routed:
        x = sample_sequences(
            model,
            alphas,
            n,
            args.max_len,
            device,
            num_timesteps_train=nt_train,
            labels=labels,
            bernoulli_scheduler=getattr(args, "bernoulli_scheduler", "loglinear"),
            generator=gen,
        )
    else:
        x = sample_sequences_simple(
            model,
            alphas,
            n,
            args.max_len,
            device,
            num_timesteps_train=nt_train,
            labels=labels,
            bernoulli_scheduler=getattr(args, "bernoulli_scheduler", "loglinear"),
            generator=gen,
        )

    strands = ids_to_strings(x)
    print(f"[samples] epoch {epoch + 1} - model generations (n={n}, seq_len={args.max_len}):")
    for i, s in enumerate(strands):
        lab = ""
        if labels is not None:
            lab = f"  label={int(labels[i].item())}"
        print(f"  gen[{i}]{lab}")
        print(f"      {_trunc(s)}")


@torch.no_grad()
def validate_routed(
    model: "RoutedDenoiser | RoutedDenoiserCNN",
    val_loader: DataLoader,
    device: torch.device,
    aux_head,
    args: "Namespace",
    *,
    epoch: int,
    global_step: int,
) -> dict[str, float]:
    """Average diffusion (and optional aux) loss on val_loader.

    Reports two diffusion-loss flavors per validation pass (same batches, same
    ``t_start`` per batch, two forward passes):

    - ``val/loss`` / ``val/diff_loss``: *with-history* loss on the training
      distribution -- views are Bernoulli corruptions of real ``x_0`` at every
      slot, exactly as in training. This is the honest training-distribution
      metric and the one the best-checkpoint criterion tracks.
    - ``val/loss_no_history`` / ``val/diff_loss_no_history``: *no-history*
      loss -- non-current view slots are replaced with the uniform simplex
      ``1/C``, matching SLM-style simple-model evaluation. Directly
      comparable to the simple model's / SLM's ``~1.3`` scale; diagnostic of
      the denoiser's capability when history is uninformative.

    Distributed-aware: each rank consumes a disjoint shard and partial sums
    are all-reduced before averaging."""
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

    val_mode = str(getattr(args, "val_new_diff_calculate", "full")).lower()
    num_timesteps = int(args.num_timesteps)
    scheduler_name = str(args.bernoulli_scheduler)

    def _nlog_p(logits: torch.Tensor, current_view: torch.Tensor, target: torch.Tensor, t_cont: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        if val_mode == "full":
            t3 = t_cont.unsqueeze(-1)
            n_t = _expected_nums(t3, scheduler=scheduler_name)
            n_t_1 = _expected_nums(t3 - 1.0 / float(num_timesteps), scheduler=scheduler_name)
            nominator = torch.clamp(n_t_1 - 1.0, min=1e-1)
            denominator = torch.clamp(n_t - 1.0, min=1e-1)
            weight = torch.clamp(nominator / denominator, min=1e-6, max=1.0 - 1e-6)
            predicted = torch.clamp(
                torch.exp(log_probs) + weight * (1.0 - torch.exp(log_probs)),
                min=1e-6,
                max=1.0 - 1e-6,
            )
            onehot = F.one_hot(target, num_classes=4).to(dtype=predicted.dtype)
            mask = current_view.to(dtype=predicted.dtype)
            bernoulli_kl = (
                (weight * torch.log(weight) + (1.0 - weight) * torch.log(1.0 - weight))
                - (weight * torch.log(predicted) + (1.0 - weight) * torch.log(1.0 - predicted))
            ) * (mask * (1.0 - onehot))
            bernoulli_kl = (bernoulli_kl * (bernoulli_kl > 0)).sum(dim=-1)
            cross_entropy_true = -torch.gather(torch.log(predicted), -1, target[:, :, None]).squeeze(-1)
            return float(num_timesteps) * (bernoulli_kl + cross_entropy_true)
        out = -torch.gather(log_probs, -1, target[:, :, None]).squeeze(-1)
        if not getattr(args, "without_T", False):
            out = float(num_timesteps) * out
        return out

    for batch in val_loader:
        x0 = batch["x0"].to(device)
        if _use_conditional_sampling_labels(args):
            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(device)
        else:
            labels = None

        t_start = int(torch.randint(0, num_timesteps, (1,), device=device).item())
        t_cont = torch.full(
            (x0.shape[0], 1),
            float(t_start + 1) / float(num_timesteps),
            device=device,
            dtype=torch.float32,
        )
        views_full = sample_all_views_bernoulli(
            x0,
            num_timesteps,
            scheduler=scheduler_name,
            generator=gen,
        )
        B, Tv, L, C = views_full.shape
        current_view = views_full[:, t_start].clone()
        views_noh = views_full.new_full((B, Tv, L, C), 1.0 / float(C))
        views_noh[:, t_start] = current_view

        labels_in = labels
        if labels is not None and getattr(args, "cond_drop_prob", 0.0) > 0:
            keep = torch.rand((x0.shape[0],), device=device) >= float(args.cond_drop_prob)
            if getattr(args, "backbone", "cnn") == "cnn":
                null_cls = int(getattr(args, "num_classes", 0))
                labels_in = torch.where(keep, labels, torch.full_like(labels, null_cls))
            elif not bool(keep.all()):
                labels_in = None

        target = x0.clamp(max=3)
        denom = float(target.shape[0] * target.shape[1])

        logits_full, _pi, h_dec, _lb, _seq_in = model(
            views_full, t_start, labels=labels_in, t_cond=float(t_cont[0, 0].item())
        )
        nlog_p_full = _nlog_p(logits_full, current_view, target, t_cont)
        diff_loss_full = nlog_p_full.float().sum() / denom

        logits_noh, _pi2, _h2, _lb2, _seq_in2 = model(
            views_noh, t_start, labels=labels_in, t_cond=float(t_cont[0, 0].item())
        )
        nlog_p_noh = _nlog_p(logits_noh, current_view, target, t_cont)
        diff_loss_noh = nlog_p_noh.float().sum() / denom

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


@torch.no_grad()
def validate_simple(
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

    for batch in val_loader:
        x0 = batch["x0"].to(device)
        if _use_conditional_sampling_labels(args):
            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(device)
        else:
            labels = None

        t_start = int(torch.randint(0, args.num_timesteps, (1,), device=device).item())
        t_cont = torch.full(
            (x0.shape[0], 1),
            float(t_start + 1) / float(args.num_timesteps),
            device=device,
            dtype=torch.float32,
        )
        x_t = corrupt_sequence_bernoulli(
            x0,
            t_cont,
            scheduler=args.bernoulli_scheduler,
            generator=gen,
        )
        t_b = t_cont.squeeze(-1)
        labels_in = labels
        if labels is not None and getattr(args, "cond_drop_prob", 0.0) > 0:
            keep = torch.rand((x0.shape[0],), device=device) >= float(args.cond_drop_prob)
            if getattr(args, "backbone", "cnn") == "cnn":
                null_cls = int(getattr(args, "num_classes", 0))
                labels_in = torch.where(keep, labels, torch.full_like(labels, null_cls))
            elif not bool(keep.all()):
                labels_in = None
        logits, h_dec = model(x_t, t_b, labels=labels_in)

        target = x0.clamp(max=3)
        log_probs = F.log_softmax(logits, dim=-1)
        val_mode = str(getattr(args, "val_new_diff_calculate", "full")).lower()
        if val_mode == "full":
            t3 = t_cont.unsqueeze(-1)
            n_t = _expected_nums(t3, scheduler=args.bernoulli_scheduler)
            n_t_1 = _expected_nums(t3 - 1.0 / float(args.num_timesteps), scheduler=args.bernoulli_scheduler)
            nominator = torch.clamp(n_t_1 - 1.0, min=1e-1)
            denominator = torch.clamp(n_t - 1.0, min=1e-1)
            weight = torch.clamp(nominator / denominator, min=1e-6, max=1.0 - 1e-6)
            predicted = torch.clamp(torch.exp(log_probs) + weight * (1.0 - torch.exp(log_probs)), min=1e-6, max=1.0 - 1e-6)
            onehot = F.one_hot(target, num_classes=4).to(dtype=predicted.dtype)
            mask = (x_t > 0).to(dtype=predicted.dtype)
            bernoulli_kl = (
                (weight * torch.log(weight) + (1.0 - weight) * torch.log(1.0 - weight))
                - (weight * torch.log(predicted) + (1.0 - weight) * torch.log(1.0 - predicted))
            ) * (mask * (1.0 - onehot))
            bernoulli_kl = (bernoulli_kl * (bernoulli_kl > 0)).sum(dim=-1)
            cross_entropy_true = -torch.gather(torch.log(predicted), -1, target[:, :, None]).squeeze(-1)
            nlog_p = float(args.num_timesteps) * (bernoulli_kl + cross_entropy_true)
        else:
            nlog_p = -torch.gather(log_probs, -1, target[:, :, None]).squeeze(-1)
            if not getattr(args, "without_T", False):
                nlog_p = float(args.num_timesteps) * nlog_p
        denom = float(target.shape[0] * target.shape[1])
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
def compute_fbd_routed(
    model: "RoutedDenoiser | RoutedDenoiserCNN",
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
    """Distributed-aware FBD: each rank embeds its shard of ``val_loader`` and generates its
    share of synthetic samples; embeddings are all-gathered before the Fréchet distance is
    computed on rank 0 (NaN on other ranks)."""
    from nonmarkovian.sample import sample_sequences

    model.eval()
    encoder = model.encoder
    use_labs = _use_conditional_sampling_labels(args)
    ddp_on, ws, rk = _ddp_world_rank()
    local_target = (int(n_samples) + ws - 1) // ws

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
        #print(str(getattr(args, "history_mode", "trajectory")))
        g = sample_sequences(
            model,
            alphas,
            int(take),
            seq_len,
            device,
            num_timesteps_train=int(args.num_timesteps),
            labels=lab,
            bernoulli_scheduler=getattr(args, "bernoulli_scheduler", "loglinear"),
            generator=gen,
            history_mode=str(getattr(args, "history_mode", "trajectory")),
        )
        if fbcnn is not None:
            gen_parts.append(fbcnn_embed_sequences(fbcnn, g))
        else:
            gen_parts.append(encoder_mean_pool_embeddings(encoder, g))
        collected += take

    real_emb_local = (
        torch.cat(real_chunks, dim=0)
        if real_chunks
        else torch.empty((0, 0), device=device)
    )
    real_emb = _gather_concat_embeddings(real_emb_local)
    if real_emb.shape[0] < 2:
        return float("nan")
    real_emb = real_emb[: int(n_samples)]

    emb_dim = real_emb.shape[1] if real_emb.ndim == 2 and real_emb.numel() else 0
    gen_emb_local = (
        torch.cat(gen_parts, dim=0)
        if gen_parts
        else torch.empty((0, emb_dim), device=device)
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
def compute_fbd_simple(
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
    """Distributed-aware FBD for the non-routed (simple) model — symmetric to
    :func:`compute_fbd_routed`."""
    from nonmarkovian.sample_simple import sample_sequences_simple

    model.eval()
    encoder = model.encoder
    use_labs = _use_conditional_sampling_labels(args)
    ddp_on, ws, rk = _ddp_world_rank()
    local_target = (int(n_samples) + ws - 1) // ws

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
        g = sample_sequences_simple(
            model,
            alphas,
            int(take),
            seq_len,
            device,
            num_timesteps_train=int(args.num_timesteps),
            labels=lab,
            bernoulli_scheduler=getattr(args, "bernoulli_scheduler", "loglinear"),
            generator=gen,
        )
        if fbcnn is not None:
            gen_parts.append(fbcnn_embed_sequences(fbcnn, g))
        else:
            gen_parts.append(encoder_mean_pool_embeddings(encoder, g))
        collected += take

    real_emb_local = (
        torch.cat(real_chunks, dim=0)
        if real_chunks
        else torch.empty((0, 0), device=device)
    )
    real_emb = _gather_concat_embeddings(real_emb_local)
    if real_emb.shape[0] < 2:
        return float("nan")
    real_emb = real_emb[: int(n_samples)]

    emb_dim = real_emb.shape[1] if real_emb.ndim == 2 and real_emb.numel() else 0
    gen_emb_local = (
        torch.cat(gen_parts, dim=0)
        if gen_parts
        else torch.empty((0, emb_dim), device=device)
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
def compute_fbd_uniform_random_baseline(
    model: torch.nn.Module,
    real_loader: DataLoader,
    device: torch.device,
    args: "Namespace",
    *,
    n_samples: int,
    seq_len: int,
    fbcnn: "CNNModel | None" = None,
    seed_offset: int = 31337,
) -> float:
    """
    Fréchet distance between embeddings of real sequences from ``real_loader`` (val or test split)
    and embeddings of i.i.d. uniform random DNA (tokens 0..3), length ``seq_len`` — a generative
    baseline without the denoiser (compare to epoch ``val/fbd`` from model samples on val).
    """
    model.eval()
    encoder = model.encoder
    real_chunks: list[torch.Tensor] = []
    collected = 0
    for batch in real_loader:
        if collected >= n_samples:
            break
        x0 = batch["x0"].to(device)
        pad = batch["mask_pad"].to(device)
        b = x0.shape[0]
        take = min(b, n_samples - collected)
        if fbcnn is not None:
            real_chunks.append(fbcnn_embed_sequences(fbcnn, x0[:take], pad[:take]))
        else:
            real_chunks.append(encoder_mean_pool_embeddings(encoder, x0[:take], pad[:take]))
        collected += take
    if collected < 2:
        return float("nan")

    real_emb = torch.cat(real_chunks, dim=0)[:n_samples]

    gen_parts: list[torch.Tensor] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + seed_offset)
    for start in range(0, n_samples, args.val_gen_batch):
        bs = min(args.val_gen_batch, n_samples - start)
        g = torch.randint(0, 4, (bs, seq_len), device=device, generator=gen, dtype=torch.long)
        if fbcnn is not None:
            gen_parts.append(fbcnn_embed_sequences(fbcnn, g))
        else:
            gen_parts.append(encoder_mean_pool_embeddings(encoder, g))
    gen_emb = torch.cat(gen_parts, dim=0)

    r = real_emb.cpu().numpy().astype(np.float64)
    g_np = gen_emb.cpu().numpy().astype(np.float64)
    return frechet_distance_np(r, g_np)
