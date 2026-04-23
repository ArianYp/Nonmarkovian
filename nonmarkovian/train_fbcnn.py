"""Train the FBCNN fly-brain enhancer classifier from scratch.

Reproduces the baseline ``clsDNAclean_cnn_1stack`` run:
  - Architecture : CNNModel(4, 81, 1, classifier=True)  — 1 stack, clean (token-id) input
  - Loss         : cross-entropy over 81 cell types
  - Optimiser    : Adam, lr=1e-3, cosine-decay to 0
  - Steps        : 450 000  (or override with --max_steps)
  - Batch size   : 128

Usage (single GPU):
    python -m nonmarkovian.train_fbcnn \\
        --dfm_enhancer auto \\
        --save FBCNN.ckpt

Usage (multi-GPU via torchrun):
    torchrun --standalone --nproc_per_node=4 -m nonmarkovian.train_fbcnn \\
        --dfm_enhancer auto --batch_size 32 --save FBCNN.ckpt

The saved file is a plain ``torch.save`` dict with keys ``state_dict``,
``global_step``, ``args``.  Load it with ``load_fbcnn_classifier``.
"""
from __future__ import annotations

import argparse
import atexit
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nonmarkovian.data import DFMEnhancerDataset, collate_pad, resolve_dfm_enhancer_root
from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.distributed_utils import (
    barrier,
    cleanup_process_group,
    setup_process_group,
    unwrap_ddp,
)
from nonmarkovian.fbcnn import CNNModel

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# cosine LR schedule (linear warm-up + cosine decay to 0)
# ---------------------------------------------------------------------------

def _cosine_lr(step: int, max_steps: int, warmup_steps: int, lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# top-1 accuracy helper
# ---------------------------------------------------------------------------

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(-1) == labels).float().mean().item())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train FBCNN fly-brain CNN classifier.")
    p.add_argument("--dfm_enhancer", type=str, default="auto",
                   help="DFM root with the_code/General/data/DeepFlyBrain_data.pkl")
    p.add_argument("--dfm_melanoma", action="store_true",
                   help="Train on DeepMEL2 (melanoma) instead of fly brain.")
    p.add_argument("--max_len", type=int, default=500,
                   help="Crop/pad sequence length (DFM sequences are 500 bp).")
    p.add_argument("--num_cnn_stacks", type=int, default=1,
                   help="CNN depth = 5 × stacks. Baseline uses 1.")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=128,
                   help="Per-GPU batch size.")
    p.add_argument("--max_steps", type=int, default=450_000,
                   help="Total optimiser steps (baseline: 450 000).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=4_000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--print_freq", type=int, default=200,
                   help="Log CE loss + accuracy every N steps.")
    p.add_argument("--val_epoch_freq", type=int, default=1,
                   help="Run validation every N epochs (default 1 = every epoch).")
    p.add_argument("--save", type=str, default="FBCNN.ckpt")
    p.add_argument("--save_freq", type=int, default=50_000,
                   help="Also save an intermediate checkpoint every N steps.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", dest="use_wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    p.add_argument("--wandb_project", type=str, default="nonmarkovian")
    p.add_argument("--wandb_run_name", type=str, default="clsDNAclean_cnn_1stack")
    args = p.parse_args()

    ddp, rank, world_size, local_rank = setup_process_group()
    if ddp and not torch.cuda.is_available():
        cleanup_process_group()
        raise SystemExit("Multi-GPU training requires CUDA.")
    if ddp:
        atexit.register(cleanup_process_group)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = resolve_device_arg(args.device)

    torch.manual_seed(args.seed + rank)
    use_wandb = bool(args.use_wandb and wandb is not None and rank == 0)

    # ------------------------------------------------------------------ data
    try:
        root = resolve_dfm_enhancer_root(args.dfm_enhancer, melanoma=args.dfm_melanoma)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    train_ds = DFMEnhancerDataset(root, "train", melanoma=args.dfm_melanoma, max_len=args.max_len)
    val_ds   = DFMEnhancerDataset(root, "val",   melanoma=args.dfm_melanoma, max_len=args.max_len)
    num_cls  = train_ds.num_classes

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    train_loader  = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_pad,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_pad,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ----------------------------------------------------------------- model
    model = CNNModel(
        alphabet_size=4,
        num_cls=num_cls,
        num_cnn_stacks=args.num_cnn_stacks,
        classifier=True,
    ).to(device)

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # time_layers exist in CNNModel but are unused when classifier=True (clean_data path).
        # find_unused_parameters=True prevents DDP from crashing on those dead weights.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    n_params = sum(p.numel() for p in unwrap_ddp(model).parameters())
    if rank == 0:
        print(f"FBCNN parameters: {n_params:,}  num_cls={num_cls}  "
              f"device={device}  world_size={world_size}")

    # -------------------------------------------------------------- optimiser
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config={**vars(args), "num_cls": num_cls, "n_params": n_params},
        )

    # ----------------------------------------------------------------- train
    global_step = 0
    best_val_acc = 0.0

    def _save(path: str | Path, note: str = "") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": {f"model.{k}": v for k, v in unwrap_ddp(model).state_dict().items()},
            "global_step": global_step,
            "epoch": epoch,
            "args": vars(args),
            "num_cls": num_cls,
            "alphabet_size": 4,
            "pytorch-lightning_version": "n/a",
        }, path)
        if rank == 0:
            print(f"  saved{note}: {path}  (step={global_step})")

    # infinite data loop — stop at max_steps
    epoch = 0
    running_loss = running_acc = running_n = 0.0

    while global_step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            x0     = batch["x0"].to(device)
            labels = batch["label"].to(device)
            t_dummy = torch.zeros(x0.shape[0], device=device)

            logits = model(x0, t_dummy)
            loss   = F.cross_entropy(logits, labels)

            _set_lr(opt, _cosine_lr(global_step, args.max_steps, args.warmup_steps, args.lr))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            running_loss += float(loss.item())
            running_acc  += _accuracy(logits.detach(), labels)
            running_n    += 1
            global_step  += 1

            if rank == 0 and global_step % args.print_freq == 0:
                avg_loss = running_loss / running_n
                avg_acc  = running_acc  / running_n
                lr_now   = opt.param_groups[0]["lr"]
                print(f"step {global_step}/{args.max_steps}  "
                      f"ce={avg_loss:.4f}  acc={avg_acc:.3f}  lr={lr_now:.2e}")
                if use_wandb:
                    wandb.log({"train/ce_loss": avg_loss,
                               "train/accuracy": avg_acc,
                               "train/lr": lr_now}, step=global_step)
                running_loss = running_acc = running_n = 0.0

            # intermediate checkpoint
            if rank == 0 and args.save_freq > 0 and global_step % args.save_freq == 0:
                stem = Path(args.save)
                _save(stem.with_stem(f"{stem.stem}_step{global_step}"),
                      note=f" (intermediate step {global_step})")

        epoch += 1

        # --------------------------------------------------------- validation
        if epoch % args.val_epoch_freq == 0 or global_step >= args.max_steps:
            if rank == 0:
                model.eval()
                val_loss = val_acc = val_n = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        x0      = batch["x0"].to(device)
                        labels  = batch["label"].to(device)
                        t_dummy = torch.zeros(x0.shape[0], device=device)
                        logits  = unwrap_ddp(model)(x0, t_dummy)
                        val_loss += float(F.cross_entropy(logits, labels).item())
                        val_acc  += _accuracy(logits, labels)
                        val_n    += 1
                val_loss /= max(val_n, 1)
                val_acc  /= max(val_n, 1)
                print(f"  epoch {epoch}  val_ce={val_loss:.4f}  val_acc={val_acc:.3f}  "
                      f"step={global_step}")
                if use_wandb:
                    wandb.log({"val/ce_loss": val_loss,
                               "val/accuracy": val_acc,
                               "epoch": epoch}, step=global_step)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    _save(args.save, note=" (best val_acc)")

    # ---------------------------------------------------------- final save
    if rank == 0:
        _save(args.save, note=" (final)")
        print(f"Training done. best_val_acc={best_val_acc:.3f}  saved to {args.save}")
        if use_wandb:
            wandb.summary["best_val_acc"] = best_val_acc
            wandb.finish()


if __name__ == "__main__":
    main()
