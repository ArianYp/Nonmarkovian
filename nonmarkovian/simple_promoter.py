"""Simple (no-router) promoter denoiser — direct SLM new_diff baseline.

Mirrors SLM's training and sampling for the promoter task exactly:
  - PromoterModel backbone (4 stacks × 5 dilated CNN blocks, time + CAGE signal)
  - Bernoulli corruption of x0 at a single random t per batch
  - Cross-entropy loss × T (or unscaled when --without_T)
  - SLM-style reverse sampling with `(x_t > 0)` Bernoulli support mask

Differences vs. ``nonmark_promoter.py``: no router, no views buffer, no candidate
mixing — every model call gets clean ``x_t`` as input.

Run example:
    torchrun --standalone --nproc_per_node=4 -m nonmarkovian.simple_promoter \\
        --epochs 200 --batch_size 128 --T 1000 --sampling_steps 128 \\
        --lr 5e-4 --weight_decay 0
"""

from __future__ import annotations

import argparse
import atexit
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ── Reuse Nonmarkovian / SLM building blocks ─────────────────────────────────
from nonmarkovian.nonmark_promoter import (
    PromoterModel,
    PromoterDatasetWrapper,
    collate_promoter,
    _load_sei,
    _get_sei_profile,
    _expected_nums,
    _sample_bernoulli,
    _EMA,
    _HAS_PROMOTER_DEPS,
    _PROMOTER_IMPORT_ERR,
)
from nonmarkovian.forward import get_xt_bernoulli
from nonmarkovian.distributed_utils import (
    barrier,
    cleanup_process_group,
    setup_process_group,
    unwrap_ddp,
)
from nonmarkovian.device_utils import cuda_is_usable, resolve_device_arg

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


# ══════════════════════════════════════════════════════════════════════════════
# Model wrapper — PromoterModel + signal threading, no router
# ══════════════════════════════════════════════════════════════════════════════

class SimpleDenoiserPromoter(nn.Module):
    """Bare PromoterModel for new_diff Bernoulli-simplex diffusion (no routing)."""

    def __init__(
        self,
        *,
        embed_dim: int = 256,
        n_hidden: int = 256,
        signal_channels: int = 1,
    ) -> None:
        super().__init__()
        self.promoter_model = PromoterModel(
            embed_dim=embed_dim,
            n_hidden=n_hidden,
            alphabet_size=4,
            signal_channels=signal_channels,
        )

    def forward(
        self,
        x_t: torch.Tensor,    # [B, L, 4] simplex
        signal: torch.Tensor, # [B, L, 1] CAGE signal
        t_cond: float,        # scalar in (0, 1]
    ) -> torch.Tensor:
        B = x_t.shape[0]
        device = x_t.device
        t_b = torch.full((B,), float(t_cond), device=device, dtype=torch.float32)
        return self.promoter_model(x_t, t_b, signal)   # [B, L, 4] logits


# ══════════════════════════════════════════════════════════════════════════════
# Reverse sampling — byte-for-byte mirror of SLM `_sample_newdiff`
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_simple_promoter(
    model:      SimpleDenoiserPromoter,
    signal:     torch.Tensor,   # [B, L, 1]
    num_steps:  int,
    device:     torch.device,
    seq_len:    int,
    vocab_size: int = 4,
    scheduler:  str = "loglinear",
) -> torch.Tensor:
    """SLM new_diff reverse sampling — no router, no history."""
    model.eval()
    B, C = signal.shape[0], int(vocab_size)
    T = int(num_steps)

    x_t = torch.full((B, seq_len, C), 1.0 / C, device=device, dtype=torch.float32)

    for i in range(1, T + 1):
        t_val = 1.0 - float(i - 1) / float(T)        # 1.0 → 1/T

        logits = model(x_t, signal, t_cond=t_val)    # [B, L, 4]

        # SLM `_new_diff_parameterization` (inference): mask outside support
        support_mask = x_t > 0
        has_any = support_mask.any(dim=-1, keepdim=True)
        support_mask = torch.where(has_any, support_mask, torch.ones_like(support_mask))
        logits = logits.masked_fill(~support_mask, torch.finfo(logits.dtype).min)

        model_prob = F.softmax(logits, dim=-1)
        prob_sum = model_prob.sum(dim=-1, keepdim=True)
        if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-4):
            model_prob = model_prob / prob_sum.clamp(min=1e-8)

        t3 = torch.full((B, 1, 1), t_val, device=device, dtype=torch.float32)
        nominator   = torch.clamp(_expected_nums(t3 - 1.0 / T, C, scheduler) - 1.0, min=1e-1)
        denominator = torch.clamp(_expected_nums(t3,            C, scheduler) - 1.0, min=1e-1)
        weight   = torch.clamp(nominator / denominator, min=0.0, max=1.0)
        predicted = torch.clamp(model_prob + weight * (1.0 - model_prob), min=0.0, max=1.0)

        # SLM Bernoulli mask: (x_t > 0)  — no one_hot(hat_x0) extension here
        sample_pred = _sample_bernoulli(predicted) & (x_t > 0)
        sample_pred_sum = sample_pred.sum(dim=-1, keepdim=True)
        fallback = F.one_hot(predicted.argmax(dim=-1), num_classes=C).to(dtype=torch.bool)
        sample_pred = torch.where(sample_pred_sum > 0, sample_pred, fallback)
        x_t = sample_pred.to(dtype=torch.float32)
        x_t = x_t / x_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # final denoising step (separate, same as SLM)
    t_last = 1.0 / float(T)
    logits_last = model(x_t, signal, t_cond=t_last)
    return logits_last.argmax(dim=-1).clamp(max=C - 1)


# ══════════════════════════════════════════════════════════════════════════════
# Validation — NLL every epoch, sp-mse every sei_epoch_freq epochs
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate_simple_promoter(
    model:           SimpleDenoiserPromoter,
    val_loader:      DataLoader,
    device:          torch.device,
    args:            argparse.Namespace,
    *,
    epoch:           int = 0,
    sei              = None,
    h3k4me3_mask:    np.ndarray | None = None,
    max_sei_batches: int = 4,
) -> dict:
    model.eval()
    T = int(args.T)
    scheduler = args.bernoulli_scheduler
    without_T = args.without_T
    num_ts    = args.num_timesteps

    total_nll, total_tokens = 0.0, 0
    sp_mse_vals: list[float] = []

    for batch_idx, batch in enumerate(val_loader):
        x0     = batch["x0"].to(device)
        signal = batch["signal"].to(device)
        B, L   = x0.shape

        # ── val NLL — single Bernoulli corruption, mirrors training ────────
        gen = torch.Generator(device=device).manual_seed(epoch * 10_000 + batch_idx)
        t_start = int(torch.randint(0, num_ts, (1,), generator=gen, device=device).item())
        t_cont  = float(t_start + 1) / float(num_ts)
        t_b     = torch.full((B, 1), t_cont, device=device, dtype=torch.float32)

        x_t = get_xt_bernoulli(x0, t_b, num_classes=4, scheduler=scheduler, generator=gen)
        logits = model(x_t, signal, t_cond=t_cont)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, x0[:, :, None]).squeeze(-1)
        if not without_T:
            nll = float(T) * nll
        total_nll    += float(nll.sum().item())
        total_tokens += B * L

        # ── SEI sp-mse  ─────────────────────────────────────────────────────
        # max_sei_batches < 0 ⇒ run on all batches (full validation set).
        sei_active = (sei is not None and h3k4me3_mask is not None
                      and (max_sei_batches < 0 or batch_idx < max_sei_batches))
        if sei_active:
            real_oh = F.one_hot(x0, num_classes=4).float()
            real_sc = _get_sei_profile(sei, h3k4me3_mask, real_oh, device)

            gen_ids = sample_simple_promoter(
                model, signal,
                num_steps=args.sampling_steps,
                device=device, seq_len=L,
                scheduler=scheduler,
            )
            gen_oh = F.one_hot(gen_ids, num_classes=4).float()
            gen_sc = _get_sei_profile(sei, h3k4me3_mask, gen_oh, device)
            sp_mse_vals.append(float(((real_sc - gen_sc) ** 2).mean()))

    metrics: dict = {
        "val/loss": total_nll / max(total_tokens, 1),
        "epoch":    epoch + 1,
    }
    if sp_mse_vals:
        metrics["val/sp_mse"] = float(np.mean(sp_mse_vals))
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Simple (no-router) PromoterModel — SLM new_diff baseline.",
    )
    # Data
    p.add_argument("--data_dir", type=str,
                   default="/lustre/scratch126/cellgen/lotfollahi/ha11/dirichlet-flow-matching/data/promoter_design")
    p.add_argument("--n_tsses",    type=int, default=100_000)
    p.add_argument("--seq_length", type=int, default=1024)
    # Training
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--ema_decay",    type=float, default=0.9999)
    # Diffusion
    p.add_argument("--T",                   type=int, default=1000)
    p.add_argument("--num_timesteps",       type=int, default=1000)
    p.add_argument("--sampling_steps",      type=int, default=100)
    p.add_argument("--bernoulli_scheduler", type=str, default="loglinear",
                   choices=("loglinear", "linear"))
    p.add_argument("--without_T", action="store_true")
    # Model
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--n_hidden",  type=int, default=256)
    # Validation / checkpointing
    p.add_argument("--val_batch_size",  type=int, default=0)
    p.add_argument("--val_epoch_freq",  type=int, default=1)
    p.add_argument("--sei_epoch_freq",  type=int, default=10)
    p.add_argument("--max_sei_batches", type=int, default=4)
    p.add_argument("--best_mse_epochs", type=int, default=50,
                   help="In the last N epochs run SEI on ALL val batches every epoch "
                        "and save a best-sp_mse checkpoint.")
    p.add_argument("--save", type=str, default="checkpoints/simple_promoter.pt")
    p.add_argument("--seed", type=int, default=0)
    # W&B
    p.add_argument("--wandb",     dest="use_wandb", action="store_true",  default=True)
    p.add_argument("--no-wandb",  dest="use_wandb", action="store_false")
    p.add_argument("--wandb_project",  type=str, default="nonmarkovian_promoter")
    p.add_argument("--wandb_run_name", type=str, default="simple")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_save_path(requested: str | Path, use_wandb: bool) -> Path:
    requested = Path(requested)
    if use_wandb and wandb is not None and getattr(wandb, "run", None) is not None:
        run_dir = getattr(wandb.run, "dir", None)
        if run_dir:
            return Path(run_dir) / requested.name
    return requested


def _train_loop(
    args:          argparse.Namespace,
    device:        torch.device,
    model:         SimpleDenoiserPromoter,
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

    # ── load Sei (rank-0 only) ────────────────────────────────────────────────
    sei, h3k4me3_mask = None, None
    if rank == 0:
        sei_path   = Path(args.data_dir) / "best.sei.model.pth.tar"
        names_path = Path(args.data_dir) / "target.sei.names"
        if (args.sei_epoch_freq > 0 and sei_path.is_file() and names_path.is_file()
                and _HAS_PROMOTER_DEPS):
            try:
                import pandas as pd
                feats        = pd.read_csv(str(names_path), sep="|", header=None)
                h3k4me3_mask = (feats[1].str.strip().values == "H3K4me3")
                sei          = _load_sei(sei_path, device)
                print(f"  SEI loaded ({h3k4me3_mask.sum()} H3K4me3 features)")
            except Exception as exc:
                print(f"  Warning: SEI load failed ({exc}); sp-mse disabled.")

    best_val, best_path = float("inf"), None
    best_mse, best_mse_path = float("inf"), None
    global_step = 0
    T = int(args.T)
    scheduler = args.bernoulli_scheduler

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch_idx, batch in enumerate(train_loader):
            x0     = batch["x0"].to(device)
            signal = batch["signal"].to(device)
            B, L   = x0.shape

            gen = torch.Generator(device=device)
            gen.manual_seed(global_step + epoch * 100_000)

            # SLM `_forward_new_diffusion`: t in {1/T, ..., 1}
            t_start = int(torch.randint(0, args.num_timesteps, (1,), device=device).item())
            t_cont  = float(t_start + 1) / float(args.num_timesteps)
            t_b     = torch.full((B, 1), t_cont, device=device, dtype=torch.float32)

            x_t = get_xt_bernoulli(x0, t_b, num_classes=4, scheduler=scheduler, generator=gen)
            logits = model(x_t, signal, t_cond=t_cont)

            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(-1, x0[:, :, None]).squeeze(-1)
            if not args.without_T:
                nll = float(T) * nll
            loss = nll.float().sum() / float(B * L)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update()

            total_loss  += float(loss.item())
            n_batches   += 1
            global_step += 1

            if use_wandb:
                wandb.log({
                    "train/loss":      float(loss.item()),
                    "train/t_start":   t_start,
                    "train/lr":        opt.param_groups[0]["lr"],
                    "train/grad_norm": float(grad_norm) if torch.is_tensor(grad_norm) else float(grad_norm),
                    "epoch":           epoch + 1,
                }, step=global_step)

        avg = total_loss / max(n_batches, 1)
        if rank == 0:
            print(f"epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}")
        if use_wandb:
            wandb.log({"train/epoch_loss_avg": avg, "epoch": epoch + 1}, step=global_step)

        # ── validation ────────────────────────────────────────────────────────
        if val_loader is not None and (epoch + 1) % args.val_epoch_freq == 0:
            m = unwrap_ddp(model)
            if ema is not None:
                ema.store(); ema.copy_to()

            in_mse_window = (args.epochs - epoch) <= args.best_mse_epochs
            run_sei_periodic = (
                args.sei_epoch_freq > 0
                and (epoch + 1) % args.sei_epoch_freq == 0
            )
            run_sei = (
                sei is not None and h3k4me3_mask is not None
                and (in_mse_window or run_sei_periodic)
            )
            # In the best-mse window: SEI on the full val set every epoch.
            # Outside the window: keep the configured cap (default 4 batches).
            sei_batches_this_epoch = -1 if in_mse_window else args.max_sei_batches
            vmetrics = validate_simple_promoter(
                m, val_loader, device, args,
                epoch=epoch,
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

                cur = float(vmetrics["val/loss"])
                if cur < best_val:
                    best_val = cur
                    save_path = _resolve_save_path(args.save, use_wandb)
                    best_path = save_path.with_name(f"{save_path.stem}.best{save_path.suffix}")
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model":            m.state_dict(),
                        "args":             vars(args),
                        "best_val_loss":    best_val,
                        "best_epoch":       epoch + 1,
                        "best_global_step": global_step,
                    }, best_path)
                    print(f"  best checkpoint → {best_path}  (val/loss={best_val:.4f})")
                    if use_wandb:
                        wandb.summary["checkpoint_best_path"]     = str(best_path.resolve())
                        wandb.summary["checkpoint_best_val_loss"] = best_val

                # Best-MSE checkpoint — only inside the last-N-epochs window,
                # using the full val-set sp_mse (representative, not biased to
                # the first 4 batches like sei_epoch_freq evaluations).
                cur_mse = vmetrics.get("val/sp_mse")
                if in_mse_window and cur_mse is not None and float(cur_mse) < best_mse:
                    best_mse = float(cur_mse)
                    save_path = _resolve_save_path(args.save, use_wandb)
                    best_mse_path = save_path.with_name(f"{save_path.stem}.best_mse{save_path.suffix}")
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

    if rank == 0:
        save_path = _resolve_save_path(args.save, use_wandb)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": unwrap_ddp(model).state_dict(), "args": vars(args)}, save_path)
        print(f"Saved final checkpoint: {save_path}")
        if best_path is not None:
            print(f"Best checkpoint:  {best_path}  (val/loss={best_val:.4f})")
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

    if not _HAS_PROMOTER_DEPS:
        raise ImportError(f"PromoterDataset deps missing: {_PROMOTER_IMPORT_ERR}")

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

    model = SimpleDenoiserPromoter(
        embed_dim=args.embed_dim,
        n_hidden=args.n_hidden,
        signal_channels=1,
    ).to(device)

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    n_params = sum(p.numel() for p in unwrap_ddp(model).parameters())
    if rank == 0:
        print(f"SimpleDenoiserPromoter: {n_params:,} parameters ({n_params / 1e6:.2f} M)")
        print(f"Dataset: {len(train_ds):,} train  {len(val_ds):,} val  "
              f"| seq_length={args.seq_length}  T={args.T}  "
              f"sampling_steps={args.sampling_steps}")
        if ddp:
            print(f"Distributed: world_size={world_size}  per-GPU batch={args.batch_size}  "
                  f"global_batch={args.batch_size * world_size}")

    if use_wandb:
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name or None,
                   config=vars(args))
        wandb.summary["model/total_parameters"] = n_params
        wandb.summary["device"]                 = str(device)
        wandb.summary["cuda_usable"]            = cuda_is_usable()
        wandb.summary["distributed"]            = ddp
        wandb.summary["world_size"]             = world_size
        if device.type == "cuda":
            wandb.summary["cuda_device_name"] = torch.cuda.get_device_name(device)

    try:
        _train_loop(args, device, model, train_loader, val_loader, use_wandb,
                    rank=rank, ddp=ddp, train_sampler=train_sampler)
    finally:
        if use_wandb:
            wandb.finish()

    if ddp:
        cleanup_process_group()


if __name__ == "__main__":
    main()
