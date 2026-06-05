"""Evaluate sp-mse on the promoter val set for a SimpleDenoiserPromoter checkpoint.

Identical procedure to SLM's validation_step: for every val batch generate
sequences via sample_simple_promoter, compute mean H3K4me3 SEI profile of real
vs generated, accumulate MSE.

Usage:
    python eval_simple_promoter_mse.py \
        --ckpt wandb/<run>/files/simple_promoter.best.pt \
        --num_batches -1            # -1 = all val batches
        --batch_size 32
        --device cuda:0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── make SLM and Nonmarkovian importable ──────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent
_SLM_ROOT = _ROOT.parent / "SLM"
for p in [str(_ROOT), str(_SLM_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from nonmarkovian.simple_promoter import (
    SimpleDenoiserPromoter,
    sample_simple_promoter,
)
from nonmarkovian.nonmark_promoter import (
    PromoterDatasetWrapper,
    collate_promoter,
    _load_sei,
    _get_sei_profile,
)


def _build_model(saved_args: dict) -> SimpleDenoiserPromoter:
    return SimpleDenoiserPromoter(
        embed_dim=saved_args.get("embed_dim", 256),
        n_hidden=saved_args.get("n_hidden", 256),
        signal_channels=1,
    )


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> float:
    device = torch.device(args.device)

    # ── load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved_args: dict = ckpt.get("args", {})
    print(f"  saved at epoch {ckpt.get('best_epoch', '?')} "
          f"| val/loss(NLL) = {ckpt.get('best_val_loss', float('nan')):.4f}")

    model = _build_model(saved_args)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters ({n_params / 1e6:.2f} M)")

    # ── SEI ───────────────────────────────────────────────────────────────────
    data_dir   = Path(args.data_dir or saved_args.get("data_dir", "."))
    sei_path   = data_dir / "best.sei.model.pth.tar"
    names_path = data_dir / "target.sei.names"

    import pandas as pd
    feats        = pd.read_csv(str(names_path), sep="|", header=None)
    h3k4me3_mask = (feats[1].str.strip().values == "H3K4me3")
    print(f"  H3K4me3 features: {h3k4me3_mask.sum()}")

    sei = _load_sei(sei_path, device)

    # ── val dataloader ────────────────────────────────────────────────────────
    seq_length = saved_args.get("seq_length", 1024)
    n_tsses    = saved_args.get("n_tsses", 100_000)
    batch_size = args.batch_size

    val_ds = PromoterDatasetWrapper(
        data_dir, split=args.split,
        seq_length=seq_length, n_tsses=n_tsses,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_promoter,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    num_batches = args.num_batches if args.num_batches > 0 else len(val_loader)
    print(f"  split={args.split}  sequences: {len(val_ds):,}  | batch_size: {batch_size}  "
          f"| evaluating {num_batches}/{len(val_loader)} batches")

    # ── sampling config ───────────────────────────────────────────────────────
    sampling_steps = args.sampling_steps or saved_args.get("sampling_steps", 100)
    scheduler      = saved_args.get("bernoulli_scheduler", "loglinear")
    print(f"  sampling_steps={sampling_steps}  scheduler={scheduler}")

    # ── evaluate — identical procedure to SLM validation_step ────────────────
    sp_mse_vals: list[float] = []

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= num_batches:
            break

        x0     = batch["x0"].to(device)       # [B, L]
        signal = batch["signal"].to(device)   # [B, L, 1]

        real_oh = F.one_hot(x0, num_classes=4).float()
        real_sc = _get_sei_profile(sei, h3k4me3_mask, real_oh, device)

        gen_ids = sample_simple_promoter(
            model, signal,
            num_steps=sampling_steps,
            device=device,
            seq_len=seq_length,
            scheduler=scheduler,
        )
        gen_oh = F.one_hot(gen_ids, num_classes=4).float()
        gen_sc = _get_sei_profile(sei, h3k4me3_mask, gen_oh, device)

        mse = float(((real_sc - gen_sc) ** 2).mean())
        sp_mse_vals.append(mse)

        print(f"  batch {batch_idx + 1}/{num_batches}  sp-mse={mse:.6f}  "
              f"(running mean={np.mean(sp_mse_vals):.6f})")

    final = float(np.mean(sp_mse_vals))
    print(f"\nFinal sp-mse: {final:.6f}  "
          f"(over {len(sp_mse_vals)} batches × {batch_size} sequences)")
    return final


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt",           type=str, required=True,
                   help="Path to .pt checkpoint (best or final).")
    p.add_argument("--data_dir",       type=str, default="",
                   help="Override data_dir (default: use value saved in checkpoint).")
    p.add_argument("--split",          type=str, default="test",
                   choices=("valid", "test"),
                   help="Which split to evaluate on. SLM training reports use 'valid'; "
                        "final held-out numbers should use 'test'.")
    p.add_argument("--num_batches",    type=int, default=-1,
                   help="Number of val batches to evaluate (-1 = all).")
    p.add_argument("--batch_size",     type=int, default=32,
                   help="Batch size for evaluation.")
    p.add_argument("--sampling_steps", type=int, default=0,
                   help="Reverse steps (0 = use checkpoint's saved value).")
    p.add_argument("--device",         type=str, default="cuda:0")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
