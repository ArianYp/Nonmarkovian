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

``--router_ablation {uniform,top1}`` runs the routed checkpoint with the router's weighting over
history states replaced at inference time (see ``nonmarkovian.router_ablations``); it is the same
flag, the same implementation and the same semantics as in ``eval_checkpoint.py``.

Add ``--dump_sequences samples_nm.txt`` to also write the generated sequences (one ACGT string
per line) for the sequence-level metrics — e.g. ``nonmarkovian.motif_metrics --sets``. The dump
reuses the samples the FBD pass already drew, so it costs nothing extra and the file is exactly
what the printed FBD scored. ``--dump_real real.txt`` writes the matching real split.

Single-GPU only (no ``torchrun`` needed).
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.eval_checkpoint import (
    _build_args_namespace,
    _build_loader,
    _build_routed_model,
    _build_simple_model,
    _dump_real,
    _suffixed,
    _write_sequences,
)
from nonmarkovian.forward import cosine_alpha_schedule
from nonmarkovian.router_ablations import ROUTER_ABLATIONS, apply_router_ablation
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
    if t in ("routed_discrete", "simple_discrete"):
        # Mirror of the guard in eval_checkpoint._detect_trainer: the architectures are shared, so
        # without this a Bernoulli checkpoint would sample through the MDLM reverse process.
        raise SystemExit(
            f"Checkpoint was trained as {t!r} (Bernoulli / ShortListing), but this is an MDLM "
            "script. Use nonmarkovian.eval_checkpoint or nonmarkovian.mind_change_slm."
        )
    state = ckpt.get("model") or {}
    has_router = any(k.startswith("W_phi") or ".W_phi" in k for k in state.keys())
    return "routed_mdlm" if has_router else "simple_mdlm"


@torch.no_grad()
def _generate_sequences_mdlm(
    trainer: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: Namespace,
    *,
    n: int,
    num_steps: int,
    epoch: int,
) -> torch.Tensor:
    """Sample ``n`` sequences the same way the MDLM FBD pass does.

    Only used when FBD is skipped; otherwise the dump reuses the FBD pass's own samples so the
    written sequences are exactly the ones the reported FBD was computed on.
    """
    from nonmarkovian.sample_mdlm import sample_sequences_mdlm
    from nonmarkovian.sample_simple_mdlm import sample_sequences_simple_mdlm
    from nonmarkovian.validation import _use_conditional_sampling_labels

    model.eval()
    use_labs = _use_conditional_sampling_labels(args)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 424242 + int(epoch) * 100003)
    seq_len = int(getattr(args, "max_len", 500))
    scheduler = str(getattr(args, "bernoulli_scheduler", "loglinear"))
    parts: list[torch.Tensor] = []
    collected = 0
    for batch in loader:
        if collected >= n:
            break
        take = min(int(batch["x0"].shape[0]), n - collected)
        labels = batch.get("label")
        lab = labels[:take].to(device) if (use_labs and labels is not None) else None
        if trainer == "routed_mdlm":
            g = sample_sequences_mdlm(
                model, num_steps, int(take), seq_len, device,
                num_timesteps_train=int(args.num_timesteps),
                labels=lab,
                guidance_scale=float(getattr(args, "guidance_scale", 0.0)),
                scheduler=scheduler,
                generator=gen,
                history_mode=str(getattr(args, "history_mode", "trajectory")),
                corruption_mode=str(getattr(args, "corruption_mode", "independent")),
                independent_threshold=float(getattr(args, "independent_threshold", 0.6)),
            )
        else:
            g = sample_sequences_simple_mdlm(
                model, num_steps, int(take), seq_len, device,
                num_timesteps_train=int(args.num_timesteps),
                labels=lab,
                guidance_scale=float(getattr(args, "guidance_scale", 0.0)),
                scheduler=scheduler,
                generator=gen,
            )
        parts.append(g.detach().to("cpu", torch.uint8))
        collected += take
    if not parts:
        raise SystemExit("--dump_sequences: the loader yielded no batches to sample against.")
    return torch.cat(parts, dim=0)[:n]


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
    p.add_argument(
        "--router_ablation",
        type=str,
        default="none",
        choices=ROUTER_ABLATIONS,
        help=(
            "Routed-only, inference-only ablation of the router's weighting over history states "
            "(see nonmarkovian.router_ablations). 'uniform' = 1/K weights (history kept, "
            "selection destroyed); 'top1' = hard argmax pick (selection kept, blending removed)."
        ),
    )
    p.add_argument(
        "--dump_sequences",
        type=str,
        default="",
        help=(
            "Write the generated sequences to this .txt (one ACGT sequence per line) for "
            "nonmarkovian.motif_metrics --sets. Reuses the FBD pass's own samples; with "
            "--fbd_no_history the uniform-history pass also lands in <stem>_no_history.txt."
        ),
    )
    p.add_argument(
        "--dump_real",
        type=str,
        default="",
        help="Also write the real split sequences to this .txt (usable as motif_metrics --real).",
    )
    p.add_argument(
        "--n_dump",
        type=int,
        default=0,
        help="How many sequences to generate when --skip_fbd is set. 0 = whole split.",
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

    # --- optional inference-time router ablation (routed checkpoints only) ---
    ablation = str(cli.router_ablation).strip().lower()
    if ablation != "none":
        if trainer != "routed_mdlm":
            raise SystemExit("--router_ablation only applies to routed_mdlm checkpoints.")
        apply_router_ablation(model, ablation)
        print(f"[eval-mdlm] router ablation active: {ablation}")

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
        f"router_ablation={ablation}  "
        f"fbcnn={'yes' if fbcnn is not None else 'no'}  "
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
    dump_path = Path(cli.dump_sequences) if cli.dump_sequences else None
    # Collect the FBD pass's own samples when a dump was asked for, so the written sequences are
    # exactly the ones the printed FBD scored (no second, differently-seeded sampling run).
    gen_seqs: list[torch.Tensor] | None = [] if dump_path is not None else None
    gen_seqs_noh: list[torch.Tensor] | None = None
    epoch = int(ck_epoch) if ck_epoch is not None else 0
    if not cli.skip_fbd:
        n_fbd = int(cli.n_fbd) if cli.n_fbd > 0 else len(loader.dataset)
        if n_fbd < 2:
            print("[eval-mdlm] fbd: skipped (need >= 2 examples).")
        else:
            tag = "fbd_fbcnn" if fbcnn is not None else "fbd"
            seq_len = int(getattr(args, "max_len", 500))
            if trainer == "routed_mdlm":
                fbd = compute_fbd_routed_mdlm(
                    model, loader, alphas_sample, device, args,
                    n_samples=n_fbd, seq_len=seq_len, epoch=epoch, fbcnn=fbcnn,
                    collect_sequences=gen_seqs,
                )
                print(f"[eval-mdlm] {tag}: {float(fbd):.4f}  (n_samples={n_fbd})")
                if cli.fbd_no_history:
                    import copy

                    args_noh = copy.copy(args)
                    args_noh.history_mode = "uniform"
                    gen_seqs_noh = [] if dump_path is not None else None
                    fbd_noh = compute_fbd_routed_mdlm(
                        model, loader, alphas_sample, device, args_noh,
                        n_samples=n_fbd, seq_len=seq_len, epoch=epoch, fbcnn=fbcnn,
                        collect_sequences=gen_seqs_noh,
                    )
                    print(f"[eval-mdlm] {tag}_no_history: {float(fbd_noh):.4f}  (n_samples={n_fbd})")
            else:
                fbd = compute_fbd_simple_mdlm(
                    model, loader, alphas_sample, device, args,
                    n_samples=n_fbd, seq_len=seq_len, epoch=epoch, fbcnn=fbcnn,
                    collect_sequences=gen_seqs,
                )
                print(f"[eval-mdlm] {tag}: {float(fbd):.4f}  (n_samples={n_fbd})")

    # --- dump sequences for downstream sequence-level metrics (motif_metrics, ...) ---
    n_dump = int(cli.n_dump) if cli.n_dump > 0 else (
        int(cli.n_fbd) if cli.n_fbd > 0 else len(loader.dataset)
    )
    if dump_path is not None:
        if gen_seqs:
            _write_sequences(dump_path, torch.cat(gen_seqs, dim=0)[:n_dump])
        else:
            # --skip_fbd (or n_fbd < 2): nothing was generated above, so sample here.
            _write_sequences(
                dump_path,
                _generate_sequences_mdlm(
                    trainer, model, loader, device, args,
                    n=n_dump, num_steps=int(alphas_sample.shape[0]), epoch=epoch,
                ),
            )
        if gen_seqs_noh:
            _write_sequences(
                _suffixed(dump_path, "_no_history"), torch.cat(gen_seqs_noh, dim=0)[:n_dump]
            )
    if cli.dump_real:
        _dump_real(loader, Path(cli.dump_real), n_dump)


if __name__ == "__main__":
    main()
