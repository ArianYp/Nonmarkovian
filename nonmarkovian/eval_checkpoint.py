"""Evaluate a saved diffusion checkpoint (routed OR simple).

Loads the model + args embedded in a checkpoint (produced by ``train.py`` or
``train_simple.py``), builds a val/test loader identical to training, and reports:

* ``val/loss`` (+ ``val/loss_no_history`` when the ckpt is routed)
* ``val/fbd`` on the chosen split

The trainer type (``routed_discrete`` vs ``simple_discrete``) is auto-detected
from the checkpoint's ``trainer`` field, falling back to a state-dict heuristic
for old files without the field.

Typical usage::

    python -m nonmarkovian.eval_checkpoint \
        --checkpoint checkpoints/simple_cnn.best.pt \
        --fbcnn_ckpt fbd.ckpt \
        --split test

Single-GPU only (no ``torchrun`` needed); the validation helpers transparently
fall back to local tensors when ``torch.distributed`` isn't initialised.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nonmarkovian.data import DFMEnhancerDataset, collate_pad, resolve_dfm_enhancer_root
from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.forward import cosine_alpha_schedule
from nonmarkovian.validation import (
    compute_fbd_routed,
    compute_fbd_simple,
    validate_routed,
    validate_simple,
)


def _detect_trainer(ckpt: dict) -> str:
    """Return ``"routed_discrete"`` or ``"simple_discrete"`` for a loaded checkpoint.

    Prefers the explicit ``trainer`` key saved by the training scripts; otherwise
    inspects the model state dict keys (routed models have ``W_phi.*``, the
    simple CNN model has ``cnn.*`` without ``W_phi``).
    """
    t = str(ckpt.get("trainer", "")).strip().lower()
    if t in ("routed_discrete", "simple_discrete"):
        return t
    state = ckpt.get("model") or {}
    keys = list(state.keys())
    has_router = any(k.startswith("W_phi") or ".W_phi" in k for k in keys)
    if has_router:
        return "routed_discrete"
    return "simple_discrete"


def _build_args_namespace(cfg: dict, overrides: dict) -> Namespace:
    """Reconstruct an argparse.Namespace with the same fields training used.

    ``overrides`` patches specific keys (e.g. ``val_gen_batch``); sensible
    defaults are filled in for fields that older checkpoints may lack.
    """
    merged = dict(cfg)
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    ns = Namespace(**merged)
    defaults = {
        "val_gen_batch": 8,
        "history_mode": "trajectory",
        "bernoulli_scheduler": "loglinear",
        "val_new_diff_calculate": "full",
        "without_T": False,
        "cond_drop_prob": 0.0,
        "aux_beta": 0.0,
        "backbone": "cnn",
        "num_classes": 0,
        "no_labels": True,
        "seed": 0,
        "max_len": 500,
        "num_timesteps": 32,
    }
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    # num_timesteps_sample defaults to num_timesteps when missing/zero.
    nts_cur = int(getattr(ns, "num_timesteps_sample", 0) or 0)
    if nts_cur <= 0:
        ns.num_timesteps_sample = int(ns.num_timesteps)
    return ns


def _build_routed_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    from nonmarkovian.model import RoutedDenoiserCNN

    backbone = str(cfg.get("backbone", "cnn")).lower()
    num_classes = int(cfg.get("num_classes", 0) or 0)
    num_labels = num_classes if num_classes > 0 else None
    max_len = int(cfg.get("max_len", 500))
    num_timesteps = int(cfg.get("num_timesteps", 32))
    router_tau = float(cfg.get("router_tau", 1.0))
    router_k = int(cfg.get("router_k", 1))
    if backbone == "cnn":
        return RoutedDenoiserCNN(
            d_model=int(cfg.get("d_model", 32)),
            max_len=max_len,
            num_timesteps=num_timesteps,
            num_labels=num_labels,
            router_tau=router_tau,
            router_k=router_k,
            num_cnn_stacks=int(cfg.get("cnn_stacks", 4)),
            router_conv_kernel=int(cfg.get("router_conv_kernel", 1)),
            router_out_channels=int(cfg.get("router_out_channels", 128)),
        ).to(device)
    cond_dim_raw = cfg.get("cond_dim", 0)
    cond_dim = int(cond_dim_raw) if cond_dim_raw else None
    if cond_dim == 0:
        cond_dim = None
    dec_layers_total = int(cfg.get("dec_layers", 6)) + int(cfg.get("enc_layers", 0))
    return RoutedDenoiser(
        d_model=int(cfg.get("d_model", 32)),
        nhead=int(cfg.get("nhead", 8)),
        dec_layers=dec_layers_total,
        dim_ff=int(cfg.get("dim_ff", 1024)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_len=max_len,
        num_timesteps=num_timesteps,
        num_labels=num_labels,
        cond_dim=cond_dim,
        router_tau=router_tau,
        router_k=router_k,
        time_freq_dim=int(cfg.get("time_freq_dim", 256)),
    ).to(device)


def _build_simple_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    from nonmarkovian.simple_model import DiscreteDenoiser, DiscreteDenoiserCNN

    backbone = str(cfg.get("backbone", "cnn")).lower()
    num_classes = int(cfg.get("num_classes", 0) or 0)
    num_labels = num_classes if num_classes > 0 else None
    max_len = int(cfg.get("max_len", 500))
    num_timesteps = int(cfg.get("num_timesteps", 32))
    if backbone == "cnn":
        return DiscreteDenoiserCNN(
            d_model=int(cfg.get("d_model", 32)),
            max_len=max_len,
            num_timesteps=num_timesteps,
            num_labels=num_labels,
            num_cnn_stacks=int(cfg.get("cnn_stacks", 4)),
        ).to(device)
    cond_dim_raw = cfg.get("cond_dim", 0)
    cond_dim = int(cond_dim_raw) if cond_dim_raw else None
    if cond_dim == 0:
        cond_dim = None
    return DiscreteDenoiser(
        d_model=int(cfg.get("d_model", 32)),
        nhead=int(cfg.get("nhead", 8)),
        dec_layers=int(cfg.get("dec_layers", 6)),
        dim_ff=int(cfg.get("dim_ff", 1024)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_len=max_len,
        num_timesteps=num_timesteps,
        num_labels=num_labels,
        cond_dim=cond_dim,
        time_freq_dim=int(cfg.get("time_freq_dim", 256)),
    ).to(device)


def _build_loader(
    cfg: dict,
    split: str,
    *,
    batch_size: int,
    dfm_root_override: str,
    melanoma_override: bool | None,
) -> DataLoader:
    dfm_arg = dfm_root_override or str(cfg.get("dfm_enhancer", "auto") or "auto")
    melanoma = bool(cfg.get("dfm_melanoma", False)) if melanoma_override is None else bool(melanoma_override)
    max_len = int(cfg.get("max_len", 500))
    root = resolve_dfm_enhancer_root(dfm_arg, melanoma=melanoma)
    if not root:
        raise SystemExit(f"Could not resolve dfm_enhancer (got {dfm_arg!r}).")
    ds = DFMEnhancerDataset(root, split, melanoma=melanoma, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad, num_workers=0)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a saved diffusion checkpoint (routed OR simple): val loss + FBD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt / .best.pt file.")
    p.add_argument("--split", type=str, default="val", choices=("val", "test"))
    p.add_argument(
        "--trainer",
        type=str,
        default="",
        choices=("", "routed_discrete", "simple_discrete"),
        help="Override auto-detected trainer type.",
    )
    p.add_argument("--dfm_enhancer", type=str, default="")
    p.add_argument(
        "--dfm_melanoma",
        dest="dfm_melanoma",
        action="store_true",
        default=None,
    )
    p.add_argument("--no_dfm_melanoma", dest="dfm_melanoma", action="store_false")
    p.add_argument("--batch_size", type=int, default=0)
    p.add_argument("--val_gen_batch", type=int, default=512)
    p.add_argument("--n_fbd", type=int, default=0, help="0 = use the whole split.")
    p.add_argument(
        "--history_mode",
        type=str,
        default="",
        choices=("", "trajectory", "uniform", "bernoulli_hat"),
        help="Routed-only override; empty = use ckpt default.",
    )
    p.add_argument("--num_timesteps_sample", type=int, default=0)
    p.add_argument("--fbcnn_ckpt", type=str, default="")
    p.add_argument("--fbcnn_num_cls", type=int, default=0)
    p.add_argument("--fbcnn_stacks", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--skip_val_loss", action="store_true")
    p.add_argument("--skip_fbd", action="store_true")
    p.add_argument(
        "--fbd_no_history",
        action="store_true",
        help=(
            "For routed models: also compute FBD with uniform history (all non-current "
            "slots set to 1/C). Directly comparable to the SLM/simple baseline."
        ),
    )
    p.add_argument(
        "--no_strict_load",
        action="store_true",
        help="Allow partial checkpoint loads (default is strict=True).",
    )
    cli = p.parse_args()

    device = resolve_device_arg(cli.device)
    ckpt_path = Path(cli.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = dict(ckpt.get("args", {}))
    if not cfg:
        raise SystemExit("Checkpoint is missing the 'args' key; cannot reconstruct the model.")
    state = ckpt.get("model")
    if state is None:
        raise SystemExit("Checkpoint missing 'model' state_dict.")
    state = dict(state)

    # Keep routed-CNN router dimensions aligned with the checkpoint tensors,
    # even when local defaults changed after training.
    w_phi = state.get("W_phi.weight")
    if isinstance(w_phi, torch.Tensor) and w_phi.ndim == 3:
        ckpt_out_channels = int(w_phi.shape[0])
        ckpt_kernel = int(w_phi.shape[2])
        if int(cfg.get("router_out_channels", ckpt_out_channels)) != ckpt_out_channels:
            print(
                "[eval] info: overriding router_out_channels "
                f"{cfg.get('router_out_channels')} -> {ckpt_out_channels} from checkpoint."
            )
            cfg["router_out_channels"] = ckpt_out_channels
        if int(cfg.get("router_conv_kernel", ckpt_kernel)) != ckpt_kernel:
            print(
                "[eval] info: overriding router_conv_kernel "
                f"{cfg.get('router_conv_kernel')} -> {ckpt_kernel} from checkpoint."
            )
            cfg["router_conv_kernel"] = ckpt_kernel

    trainer = cli.trainer.strip().lower() or _detect_trainer(ckpt)
    if trainer not in ("routed_discrete", "simple_discrete"):
        raise SystemExit(f"Unknown trainer type: {trainer!r}")

    # --- args Namespace with CLI overrides ---
    overrides: dict = {}
    if cli.dfm_enhancer:
        overrides["dfm_enhancer"] = cli.dfm_enhancer
    if cli.dfm_melanoma is not None:
        overrides["dfm_melanoma"] = bool(cli.dfm_melanoma)
    if cli.val_gen_batch > 0:
        overrides["val_gen_batch"] = int(cli.val_gen_batch)
    if cli.num_timesteps_sample > 0:
        overrides["num_timesteps_sample"] = int(cli.num_timesteps_sample)
    if cli.history_mode:
        overrides["history_mode"] = cli.history_mode
    if cli.fbcnn_num_cls > 0:
        overrides["fbcnn_num_cls"] = int(cli.fbcnn_num_cls)
    if cli.fbcnn_stacks > 0:
        overrides["fbcnn_stacks"] = int(cli.fbcnn_stacks)
    if cli.seed >= 0:
        overrides["seed"] = int(cli.seed)
    args = _build_args_namespace(cfg, overrides)

    # --- model ---
    if trainer == "routed_discrete":
        model = _build_routed_model(cfg, device)
        model.num_timesteps = cli.num_timesteps_sample
    else:
        model = _build_simple_model(cfg, device)
    strict_load = not bool(cli.no_strict_load)
    if "state_embed.weight" not in state and hasattr(model, "state_embed"):
        state_embed = getattr(model, "state_embed")
        if hasattr(state_embed, "weight"):
            state["state_embed.weight"] = torch.zeros_like(state_embed.weight)
            print("[eval] info: checkpoint missing state_embed.weight; initialized it to zeros for loading.")

    load_info = model.load_state_dict(state, strict=strict_load)
    if not strict_load:
        missing = getattr(load_info, "missing_keys", []) or []
        unexpected = getattr(load_info, "unexpected_keys", []) or []
        if missing:
            print(f"[eval] warning: {len(missing)} missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"[eval] warning: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    model.eval()

    # --- alphas for the reverse process ---
    alphas_sample = ckpt.get("alphas_sample")
    nts = int(args.num_timesteps_sample)
    if alphas_sample is None or alphas_sample.shape[0] != nts:
        alphas_sample = cosine_alpha_schedule(nts, device=device)
    else:
        alphas_sample = alphas_sample.to(device)

    # --- data loader ---
    batch_size = int(cli.batch_size) if cli.batch_size > 0 else int(cfg.get("val_batch_size") or cfg.get("batch_size") or 8)
    batch_size = 64
    loader = _build_loader(
        cfg,
        cli.split,
        batch_size=batch_size,
        dfm_root_override=cli.dfm_enhancer,
        melanoma_override=cli.dfm_melanoma,
    )

    # --- optional FBCNN classifier for FBD embeddings ---
    fbcnn = None
    fbcnn_path = cli.fbcnn_ckpt.strip() or str(cfg.get("fbcnn_ckpt", "") or "").strip()
    if fbcnn_path and not cli.skip_fbd:
        from nonmarkovian.fbcnn import load_fbcnn_classifier

        fbcnn = load_fbcnn_classifier(
            fbcnn_path,
            device,
            num_cls=int(cli.fbcnn_num_cls or 0),
            num_cnn_stacks=int(cli.fbcnn_stacks or 0),
        )

    ck_best = ckpt.get("best_val_loss")
    ck_best_noh = ckpt.get("best_val_loss_no_history")
    ck_epoch = ckpt.get("best_epoch")
    print(f"[eval] checkpoint: {ckpt_path.resolve()}  trainer={trainer}")
    if ck_best is not None:
        line = f"[eval] checkpoint metadata: best_val_loss={float(ck_best):.4f}"
        if ck_best_noh is not None:
            line += f"  best_val_loss_no_history={float(ck_best_noh):.4f}"
        if ck_epoch is not None:
            line += f"  best_epoch={int(ck_epoch)}"
        print(line)
    print(
        f"[eval] split={cli.split}  n={len(loader.dataset)}  batch_size={batch_size}  "
        f"num_timesteps={getattr(args, 'num_timesteps', None)}  "
        f"num_timesteps_sample={args.num_timesteps_sample}  "
        f"history_mode={getattr(args, 'history_mode', 'n/a') if trainer == 'routed_discrete' else 'n/a'}  "
        f"fbcnn={'yes' if fbcnn is not None else 'no'}"
    )

    # --- validation loss ---
    if not cli.skip_val_loss:
        if trainer == "routed_discrete":
            vmetrics = validate_routed(
                model, loader, device, None, args,
                epoch=int(ck_epoch) if ck_epoch is not None else 0,
                global_step=0,
            )
        else:
            vmetrics = validate_simple(
                model, loader, device, None, args,
                epoch=int(ck_epoch) if ck_epoch is not None else 0,
                global_step=0,
            )
        print("[eval] validation metrics:")
        for k in ("val/loss", "val/diff_loss", "val/loss_no_history", "val/diff_loss_no_history", "val/aux_loss"):
            if k in vmetrics:
                print(f"  {k}: {vmetrics[k]:.4f}")

    # --- FBD ---
    if not cli.skip_fbd:
        n_fbd = int(cli.n_fbd) if cli.n_fbd > 0 else len(loader.dataset)
        if n_fbd < 2:
            print("[eval] fbd: skipped (need >= 2 examples).")
        else:
            tag = "fbd_fbcnn" if fbcnn is not None else "fbd"
            if trainer == "routed_discrete":
                fbd = compute_fbd_routed(
                    model, loader, alphas_sample, device, args,
                    n_samples=n_fbd,
                    seq_len=int(getattr(args, "max_len", 500)),
                    epoch=int(ck_epoch) if ck_epoch is not None else 0,
                    fbcnn=fbcnn,
                )
                print(f"[eval] {tag}: {float(fbd):.4f}  (n_samples={n_fbd})")

                if cli.fbd_no_history:
                    # Second pass: uniform history — all non-current slots set to 1/C.
                    # Comparable to SLM/simple model (no history information).
                    import copy
                    args_noh = copy.copy(args)
                    args_noh.history_mode = "uniform"
                    fbd_noh = compute_fbd_routed(
                        model, loader, alphas_sample, device, args_noh,
                        n_samples=n_fbd,
                        seq_len=int(getattr(args, "max_len", 500)),
                        epoch=int(ck_epoch) if ck_epoch is not None else 0,
                        fbcnn=fbcnn,
                    )
                    print(f"[eval] {tag}_no_history: {float(fbd_noh):.4f}  (n_samples={n_fbd})")
            else:
                fbd = compute_fbd_simple(
                    model, loader, alphas_sample, device, args,
                    n_samples=n_fbd,
                    seq_len=int(getattr(args, "max_len", 500)),
                    epoch=int(ck_epoch) if ck_epoch is not None else 0,
                    fbcnn=fbcnn,
                )
                print(f"[eval] {tag}: {float(fbd):.4f}  (n_samples={n_fbd})")


if __name__ == "__main__":
    main()
