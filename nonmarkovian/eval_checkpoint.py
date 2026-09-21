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

Add ``--dump_sequences samples_nm.txt`` to also write the generated sequences (one ACGT string
per line) for the sequence-level metrics — e.g. ``nonmarkovian.motif_metrics --sets``. The dump
reuses the samples the FBD pass already drew, so it costs nothing extra and the file is exactly
what the printed FBD scored. ``--dump_real real.txt`` writes the matching real split.

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
    if t in ("routed_mdlm", "simple_mdlm"):
        # Both families use the same architecture, so an MDLM checkpoint loads cleanly here and
        # would then be sampled with the Bernoulli/ShortListing reverse process -- wrong process,
        # no error, silently garbage FBD. Refuse instead of falling through to the heuristic.
        raise SystemExit(
            f"Checkpoint was trained as {t!r} (masked / absorbing MDLM), but this is a "
            "Bernoulli / ShortListing script. Use the MDLM twin: "
            "nonmarkovian.eval_checkpoint_mdlm or nonmarkovian.mind_change_mdlm."
        )
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


def _write_sequences(path: Path, ids: torch.Tensor, mask_pad: torch.Tensor | None = None) -> int:
    """Write ``[N, L]`` token ids as one ACGT string per line — the format ``motif_metrics``
    (and any of the sequence-level metrics) reads. ``mask_pad`` trims padding, so real
    sequences come out at their true lengths rather than padded with 'A'."""
    from nonmarkovian.sample import ids_to_strings

    seqs = [s for s in ids_to_strings(ids.cpu(), mask_pad.cpu() if mask_pad is not None else None) if s]
    if not seqs:
        print(f"[eval] warning: nothing to dump to {path}; skipped.")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(seqs) + "\n")
    lens = [len(s) for s in seqs]
    # motif_metrics.read_sequences truncates every set to its shortest line, so a ragged dump
    # silently shortens the whole comparison -- worth seeing the spread here.
    print(f"[eval] dumped {len(seqs)} sequences (len {min(lens)}-{max(lens)}) -> {path}")
    return len(seqs)


def _suffixed(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix or '.txt'}")


@torch.no_grad()
def _generate_sequences(
    trainer: str,
    model: torch.nn.Module,
    loader: DataLoader,
    alphas: torch.Tensor,
    device: torch.device,
    args: Namespace,
    *,
    n: int,
    epoch: int,
) -> torch.Tensor:
    """Sample ``n`` sequences the same way the FBD pass does.

    Only used when FBD is skipped; otherwise the dump reuses the FBD pass's own samples so the
    written sequences are exactly the ones the reported FBD was computed on.
    """
    from nonmarkovian.validation import _use_conditional_sampling_labels

    model.eval()
    use_labs = _use_conditional_sampling_labels(args)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 424242 + int(epoch) * 100003)
    seq_len = int(getattr(args, "max_len", 500))
    parts: list[torch.Tensor] = []
    collected = 0
    for batch in loader:
        if collected >= n:
            break
        take = min(int(batch["x0"].shape[0]), n - collected)
        labels = batch.get("label")
        lab = labels[:take].to(device) if (use_labs and labels is not None) else None
        if trainer == "routed_discrete":
            from nonmarkovian.sample import sample_sequences

            g = sample_sequences(
                model, alphas, take, seq_len, device,
                num_timesteps_train=int(args.num_timesteps),
                labels=lab,
                guidance_scale=float(getattr(args, "guidance_scale", 0.0)),
                bernoulli_scheduler=getattr(args, "bernoulli_scheduler", "loglinear"),
                generator=gen,
                history_mode=str(getattr(args, "history_mode", "trajectory")),
                corruption_mode=str(getattr(args, "corruption_mode", "trajectory")),
            )
        else:
            from nonmarkovian.sample_simple import sample_sequences_simple

            g = sample_sequences_simple(
                model, alphas, take, seq_len, device,
                num_timesteps_train=int(args.num_timesteps),
                labels=lab,
                guidance_scale=float(getattr(args, "guidance_scale", 0.0)),
                bernoulli_scheduler=getattr(args, "bernoulli_scheduler", "loglinear"),
                generator=gen,
            )
        parts.append(g.detach().to("cpu", torch.uint8))
        collected += take
    if not parts:
        raise SystemExit("--dump_sequences: the loader yielded no batches to sample against.")
    return torch.cat(parts, dim=0)[:n]


def _dump_real(loader: DataLoader, path: Path, n: int) -> None:
    chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
    got = 0
    for batch in loader:
        if got >= n:
            break
        take = min(int(batch["x0"].shape[0]), n - got)
        chunks.append((batch["x0"][:take], batch["mask_pad"][:take]))
        got += take
    if not chunks:
        raise SystemExit("--dump_real: the loader yielded no batches.")
    # Batches are padded to their own max length, so pad every chunk out to the global width
    # before concatenating; mask_pad still marks the filler and _write_sequences trims it.
    width = max(int(x.shape[1]) for x, _ in chunks)
    xs, ms = [], []
    for x, m in chunks:
        pad = width - int(x.shape[1])
        xs.append(torch.nn.functional.pad(x, (0, pad), value=0))
        ms.append(torch.nn.functional.pad(m, (0, pad), value=True))
    _write_sequences(path, torch.cat(xs, dim=0), torch.cat(ms, dim=0))


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
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help=(
            "Classifier-free guidance scale w applied at sampling. "
            "logits_guided = (1+w)*logits_cond - w*logits_uncond. "
            "0 = pure conditional (uses real labels). -1 = pure unconditional. "
            "Typical CFG range 1-3. Requires the checkpoint to have been trained "
            "WITHOUT --no_labels (so the conditional embedding rows are trained)."
        ),
    )
    p.add_argument(
        "--bias", type=float, default=-1.0,
        help="Routed-only: additive floor on the corrector-phase re-activation rate in "
             "sample.py. <0 = use the sampler's own hardcoded default.",
    )
    p.add_argument(
        "--router_ablation",
        type=str,
        default="none",
        choices=("none", "uniform", "top1"),
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
    overrides["guidance_scale"] = float(cli.guidance_scale)
    if cli.bias >= 0.0:
        overrides["bias"] = float(cli.bias)
    args = _build_args_namespace(cfg, overrides)
    print('trainer', trainer)
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

    ablation = str(cli.router_ablation).strip().lower()
    if ablation != "none":
        if trainer != "routed_discrete":
            raise SystemExit("--router_ablation only applies to routed_discrete checkpoints.")
        from nonmarkovian.router_ablations import apply_router_ablation

        apply_router_ablation(model, ablation)
        print(f"[eval] router ablation active: {ablation}")

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
        f"guidance_scale={float(args.guidance_scale)}  "
        f"bias={getattr(args, 'bias', 'default')}  "
        f"router_ablation={ablation}  "
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
    dump_path = Path(cli.dump_sequences) if cli.dump_sequences else None
    # Collect the FBD pass's own samples when a dump was asked for, so the written sequences are
    # exactly the ones the printed FBD scored (no second, differently-seeded sampling run).
    gen_seqs: list[torch.Tensor] | None = [] if dump_path is not None else None
    gen_seqs_noh: list[torch.Tensor] | None = None
    epoch = int(ck_epoch) if ck_epoch is not None else 0
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
                    epoch=epoch,
                    fbcnn=fbcnn,
                    collect_sequences=gen_seqs,
                )
                print(f"[eval] {tag}: {float(fbd):.4f}  (n_samples={n_fbd})")

                if cli.fbd_no_history:
                    # Second pass: uniform history — all non-current slots set to 1/C.
                    # Comparable to SLM/simple model (no history information).
                    import copy
                    args_noh = copy.copy(args)
                    args_noh.history_mode = "uniform"
                    gen_seqs_noh = [] if dump_path is not None else None
                    fbd_noh = compute_fbd_routed(
                        model, loader, alphas_sample, device, args_noh,
                        n_samples=n_fbd,
                        seq_len=int(getattr(args, "max_len", 500)),
                        epoch=epoch,
                        fbcnn=fbcnn,
                        collect_sequences=gen_seqs_noh,
                    )
                    print(f"[eval] {tag}_no_history: {float(fbd_noh):.4f}  (n_samples={n_fbd})")
            else:
                fbd = compute_fbd_simple(
                    model, loader, alphas_sample, device, args,
                    n_samples=n_fbd,
                    seq_len=int(getattr(args, "max_len", 500)),
                    epoch=epoch,
                    fbcnn=fbcnn,
                    collect_sequences=gen_seqs,
                )
                print(f"[eval] {tag}: {float(fbd):.4f}  (n_samples={n_fbd})")

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
                _generate_sequences(
                    trainer, model, loader, alphas_sample, device, args,
                    n=n_dump, epoch=epoch,
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
