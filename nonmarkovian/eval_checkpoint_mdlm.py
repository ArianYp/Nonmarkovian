"""Evaluate a saved MDLM (masked / absorbing) checkpoint (routed OR simple).

MDLM twin of ``eval_checkpoint.py``. Loads the model + args embedded in a checkpoint produced by
``train_mdlm.py`` (``trainer='routed_mdlm'``) or ``train_simple_mdlm.py``
(``trainer='simple_mdlm'``), builds a val/test loader identical to training, and reports:

* ``val/loss`` (+ ``val/loss_no_history`` when the ckpt is routed) — the MDLM NELBO
* ``val/fbd`` on the chosen split, using the MDLM ancestral samplers

Model construction and the data loader are reused verbatim from ``eval_checkpoint.py`` (the
architecture is identical); only the trainer-type detection, the validation-loss function, and the
FBD function are MDLM-specific.

Typical usage::

    python -m nonmarkovian.eval_checkpoint_mdlm \
        --checkpoint checkpoints/simple_mdlm.best_fbd.pt \
        --fbcnn_ckpt fbd.ckpt \
        --split test

Single-GPU only (no ``torchrun`` needed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.eval_checkpoint import (
    _build_args_namespace,
    _build_loader,
    _build_routed_model,
    _build_simple_model,
)
from nonmarkovian.forward import cosine_alpha_schedule
from nonmarkovian.validation_mdlm import (
    compute_fbd_routed_mdlm,
    compute_fbd_simple_mdlm,
    validate_routed_mdlm,
    validate_simple_mdlm,
)


def _detect_trainer_mdlm(ckpt: dict) -> str:
    """Return ``"routed_mdlm"`` or ``"simple_mdlm"`` for a loaded checkpoint.

    Prefers the explicit ``trainer`` key saved by the MDLM training scripts; otherwise inspects
    the state dict (routed models have ``W_phi.*``)."""
    t = str(ckpt.get("trainer", "")).strip().lower()
    if t in ("routed_mdlm", "simple_mdlm"):
        return t
    state = ckpt.get("model") or {}
    has_router = any(k.startswith("W_phi") or ".W_phi" in k for k in state.keys())
    return "routed_mdlm" if has_router else "simple_mdlm"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a saved MDLM checkpoint (routed OR simple): val loss + FBD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt / .best_fbd.pt file.")
    p.add_argument("--split", type=str, default="val", choices=("val", "test"))
    p.add_argument(
        "--trainer",
        type=str,
        default="",
        choices=("", "routed_mdlm", "simple_mdlm"),
        help="Override auto-detected trainer type.",
    )
    p.add_argument("--dfm_enhancer", type=str, default="")
    p.add_argument("--dfm_melanoma", dest="dfm_melanoma", action="store_true", default=None)
    p.add_argument("--no_dfm_melanoma", dest="dfm_melanoma", action="store_false")
    p.add_argument("--batch_size", type=int, default=0)
    p.add_argument("--val_gen_batch", type=int, default=512)
    p.add_argument("--n_fbd", type=int, default=0, help="0 = use the whole split.")
    p.add_argument(
        "--history_mode",
        type=str,
        default="",
        choices=("", "trajectory", "uniform"),
        help="Routed-only override; empty = use ckpt default.",
    )
    p.add_argument(
        "--corruption_mode",
        type=str,
        default="",
        choices=("", "independent", "trajectory"),
        help="Reverse-step constraint for routed sampling; empty = use ckpt value.",
    )
    p.add_argument(
        "--independent_threshold",
        type=float,
        default=-1.0,
        help="Fraction of steps after which carry-over is dropped (independent mode); <0 = ckpt value.",
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
        help="For routed models: also compute FBD with uniform history (non-current slots -> 1/C).",
    )
    p.add_argument(
        "--no_strict_load",
        action="store_true",
        help="Allow partial checkpoint loads (default is strict=True).",
    )
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale w applied at sampling (0 = pure conditional).",
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

    # Keep routed-CNN router dimensions aligned with the checkpoint tensors.
    w_phi = state.get("W_phi.weight")
    if isinstance(w_phi, torch.Tensor) and w_phi.ndim == 3:
        cfg["router_out_channels"] = int(w_phi.shape[0])
        cfg["router_conv_kernel"] = int(w_phi.shape[2])

    trainer = cli.trainer.strip().lower() or _detect_trainer_mdlm(ckpt)
    if trainer not in ("routed_mdlm", "simple_mdlm"):
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
    if cli.corruption_mode:
        overrides["corruption_mode"] = cli.corruption_mode
    if cli.independent_threshold >= 0.0:
        overrides["independent_threshold"] = float(cli.independent_threshold)
    if cli.fbcnn_num_cls > 0:
        overrides["fbcnn_num_cls"] = int(cli.fbcnn_num_cls)
    if cli.fbcnn_stacks > 0:
        overrides["fbcnn_stacks"] = int(cli.fbcnn_stacks)
    if cli.seed >= 0:
        overrides["seed"] = int(cli.seed)
    overrides["guidance_scale"] = float(cli.guidance_scale)
    args = _build_args_namespace(cfg, overrides)

    # --- model (identical architecture to the Bernoulli builders) ---
    if trainer == "routed_mdlm":
        model = _build_routed_model(cfg, device)
        model.num_timesteps = int(args.num_timesteps_sample)
    else:
        model = _build_simple_model(cfg, device)
    strict_load = not bool(cli.no_strict_load)
    load_info = model.load_state_dict(state, strict=strict_load)
    if not strict_load:
        missing = getattr(load_info, "missing_keys", []) or []
        unexpected = getattr(load_info, "unexpected_keys", []) or []
        if missing:
            print(f"[eval] warning: {len(missing)} missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"[eval] warning: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    model.eval()

    # --- alphas for the reverse process (only its length matters for MDLM: = #reverse steps) ---
    nts = int(args.num_timesteps_sample)
    alphas_sample = ckpt.get("alphas_sample")
    if alphas_sample is None or alphas_sample.shape[0] != nts:
        alphas_sample = cosine_alpha_schedule(nts, device=device)
    else:
        alphas_sample = alphas_sample.to(device)

    # --- data loader ---
    batch_size = int(cli.batch_size) if cli.batch_size > 0 else int(
        cfg.get("val_batch_size") or cfg.get("batch_size") or 64
    )
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

    ck_epoch = ckpt.get("best_fbd_epoch") or ckpt.get("best_epoch")
    print(f"[eval-mdlm] checkpoint: {ckpt_path.resolve()}  trainer={trainer}")
    print(
        f"[eval-mdlm] split={cli.split}  n={len(loader.dataset)}  batch_size={batch_size}  "
        f"num_timesteps={getattr(args, 'num_timesteps', None)}  "
        f"num_timesteps_sample={args.num_timesteps_sample}  "
        f"scheduler={getattr(args, 'bernoulli_scheduler', 'loglinear')}  "
        f"history_mode={getattr(args, 'history_mode', 'n/a') if trainer == 'routed_mdlm' else 'n/a'}  "
        f"guidance_scale={float(args.guidance_scale)}  "
        f"fbcnn={'yes' if fbcnn is not None else 'no'}"
        f"trainer={trainer}"
    )

    # --- validation loss (MDLM NELBO) ---
    if not cli.skip_val_loss:
        if trainer == "routed_mdlm":
            vmetrics = validate_routed_mdlm(
                model, loader, device, None, args,
                epoch=int(ck_epoch) if ck_epoch is not None else 0, global_step=0,
            )
        else:
            vmetrics = validate_simple_mdlm(
                model, loader, device, None, args,
                epoch=int(ck_epoch) if ck_epoch is not None else 0, global_step=0,
            )
        print("[eval-mdlm] validation metrics:")
        for k in ("val/loss", "val/diff_loss", "val/loss_no_history", "val/diff_loss_no_history"):
            if k in vmetrics:
                print(f"  {k}: {vmetrics[k]:.4f}")

    # --- FBD (MDLM ancestral sampling) ---
    if not cli.skip_fbd:
        n_fbd = int(cli.n_fbd) if cli.n_fbd > 0 else len(loader.dataset)
        if n_fbd < 2:
            print("[eval-mdlm] fbd: skipped (need >= 2 examples).")
        else:
            tag = "fbd_fbcnn" if fbcnn is not None else "fbd"
            seq_len = int(getattr(args, "max_len", 500))
            epoch = int(ck_epoch) if ck_epoch is not None else 0
            if trainer == "routed_mdlm":
                fbd = compute_fbd_routed_mdlm(
                    model, loader, alphas_sample, device, args,
                    n_samples=n_fbd, seq_len=seq_len, epoch=epoch, fbcnn=fbcnn,
                )
                print(f"[eval-mdlm] {tag}: {float(fbd):.4f}  (n_samples={n_fbd})")
                if cli.fbd_no_history:
                    import copy

                    args_noh = copy.copy(args)
                    args_noh.history_mode = "uniform"
                    fbd_noh = compute_fbd_routed_mdlm(
                        model, loader, alphas_sample, device, args_noh,
                        n_samples=n_fbd, seq_len=seq_len, epoch=epoch, fbcnn=fbcnn,
                    )
                    print(f"[eval-mdlm] {tag}_no_history: {float(fbd_noh):.4f}  (n_samples={n_fbd})")
            else:
                fbd = compute_fbd_simple_mdlm(
                    model, loader, alphas_sample, device, args,
                    n_samples=n_fbd, seq_len=seq_len, epoch=epoch, fbcnn=fbcnn,
                )
                print(f"[eval-mdlm] {tag}: {float(fbd):.4f}  (n_samples={n_fbd})")


if __name__ == "__main__":
    main()
