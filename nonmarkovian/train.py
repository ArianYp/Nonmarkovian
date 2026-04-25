"""Train non-Markovian routed discrete diffusion on DFM enhancer pickles (Zenodo)."""

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
from nonmarkovian.device_utils import cuda_is_usable, resolve_device_arg
from nonmarkovian.distributed_utils import (
    barrier,
    cleanup_process_group,
    setup_process_group,
    unwrap_ddp,
)
from nonmarkovian.forward import cosine_alpha_schedule, sample_all_views_bernoulli
from nonmarkovian.model import ActivityAuxHead, RoutedDenoiser, RoutedDenoiserCNN
from nonmarkovian.train_timing import tic, toc_ms
from nonmarkovian.validation import (
    _use_conditional_sampling_labels,
    chance_validation_baselines,
    compute_fbd_routed,
    compute_fbd_uniform_random_baseline,
    print_epoch_diffusion_dna_samples,
    print_val_and_random_dna_preview,
    validate_routed,
)

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[misc, assignment]


def _to_float(x: torch.Tensor | float) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    return float(x)


def _resolve_save_path(requested: str | Path, use_wandb: bool) -> Path:
    """Resolve a checkpoint save path.

    When a W&B run is active, route checkpoints into ``wandb.run.dir``
    (``.../wandb/run-<timestamp>-<id>/files``) so they are colocated with
    the run's metadata, logs, and are picked up by W&B's file-sync.
    The original basename of ``requested`` is preserved. If W&B is not
    active (or ``wandb`` is unavailable), return the requested path as-is.
    """
    requested = Path(requested)
    if use_wandb and wandb is not None and getattr(wandb, "run", None) is not None:
        run_dir = getattr(wandb.run, "dir", None)
        if run_dir:
            return Path(run_dir) / requested.name
    return requested


class _EMA:
    """Simple parameter EMA with temporary swap for evaluation (matches train_simple._EMA)."""

    def __init__(self, params: list[torch.nn.Parameter], decay: float):
        self.params = [p for p in params if p.requires_grad]
        self.decay = float(decay)
        self.shadow = [p.detach().clone() for p in self.params]
        self.backup: list[torch.Tensor] | None = None

    @torch.no_grad()
    def update(self) -> None:
        one_minus = 1.0 - self.decay
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=one_minus)

    @torch.no_grad()
    def store(self) -> None:
        self.backup = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def copy_to(self) -> None:
        for p, s in zip(self.params, self.shadow):
            p.data.copy_(s)

    @torch.no_grad()
    def restore(self) -> None:
        if self.backup is None:
            return
        for p, b in zip(self.params, self.backup):
            p.data.copy_(b)
        self.backup = None


def _parse_train_args() -> argparse.Namespace:
    """Parse CLI: DiT-only hyperparameters are accepted only when ``--backbone dit``."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "DiT-only flags (--nhead, --dec_layers, --dim_ff, --dropout, --cond_dim, --time_freq_dim) "
            "are only valid with --backbone dit; omit them for the default CNN backbone."
        ),
    )
    p.add_argument(
        "--dfm_enhancer",
        type=str,
        default="auto",
        help="DFM Zenodo enhancer root (directory with the_code/). Use data_dfm, auto, or an absolute path.",
    )
    p.add_argument(
        "--dfm_melanoma",
        action="store_true",
        help="Load DeepMEL2 (melanoma) instead of fly brain.",
    )
    p.add_argument(
        "--max_len",
        type=int,
        default=500,
        help="Pad/cap sequence length for train + val data and for FBD sampling (single knob).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Train batch size per GPU. Global batch = this × world size when using torchrun.",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay for evaluation-time weight averaging (0 disables EMA).",
    )
    p.add_argument("--num_timesteps", type=int, default=32)
    p.add_argument(
        "--bernoulli_scheduler",
        type=str,
        default="loglinear",
        choices=("loglinear", "linear"),
        help="SLM-style Bernoulli corruption scheduler for routed history views.",
    )
    p.add_argument(
        "--without_T",
        action="store_true",
        help="Do not scale new_diff loss by T (matches SLM `training.without_T`).",
    )
    p.add_argument(
        "--cond_drop_prob",
        type=float,
        default=0.3,
        help="Classifier-free label drop probability (SLM new_diff uses 0.3).",
    )
    p.add_argument(
        "--val_new_diff_calculate",
        type=str,
        default="full",
        choices=("full", "training"),
        help="Validation new_diff metric: SLM-style inference `full` or training-style NLL.",
    )
    p.add_argument(
        "--val_no_history",
        action="store_true",
        default=False,
        help="DEPRECATED / no-op. validate_routed now always logs both "
        "val/loss (with real-x0 history, training distribution) and "
        "val/loss_no_history (uniform-history, SLM-comparable). The "
        "best-checkpoint criterion uses val/loss (with history).",
    )
    p.add_argument(
        "--no_val_no_history",
        dest="val_no_history",
        action="store_false",
        help="DEPRECATED / no-op (see --val_no_history).",
    )
    p.add_argument(
        "--history_mode",
        type=str,
        default="trajectory",
        choices=("trajectory", "uniform", "bernoulli_hat"),
        help=(
            "History mode for routed reverse sampling (FBD / previews). 'trajectory' (default) "
            "feeds the model the running x_t simplices it just visited; 'uniform' keeps non-current "
            "slots at 1/C (matches --val_no_history validation); 'bernoulli_hat' is the legacy "
            "Bernoulli-of-hat_x0 mode."
        ),
    )
    p.add_argument(
        "--num_timesteps_sample",
        type=int,
        default=0,
        help="Reverse steps for FBD / DNA preview / checkpoint sampling (0 = same as --num_timesteps). "
        "Training views always use --num_timesteps.",
    )
    p.add_argument(
        "--router_topk",
        type=int,
        default=0,
        help="Deprecated; use --router_k (this flag is ignored)",
    )
    p.add_argument(
        "--router_k",
        type=int,
        default=1,
        help="Ignored (routing mixes all future views with Gumbel-Softmax / softmax); kept for old configs",
    )
    p.add_argument(
        "--router_tau",
        type=float,
        default=1.0,
        help="Boltzmann / Gumbel-Softmax temperature τ (smaller → sharper routing)",
    )
    p.add_argument(
        "--router_lambda_bal",
        type=float,
        default=0.1,
        help="Weight λ for load-balancing loss (Switch-style; 0 = off)",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="cnn",
        choices=("dit", "cnn"),
        help='Denoiser backbone: "cnn" matches SLM enhancer CNNModel; "dit" is the prior DiT stack.',
    )
    p.add_argument(
        "--cnn_stacks",
        type=int,
        default=4,
        help="When backbone=cnn: CNNModel num_cnn_stacks (5 conv layers per stack; SLM fly enhancer uses 4).",
    )
    p.add_argument(
        "--router_conv_kernel",
        type=int,
        default=1,
        help="When backbone=cnn: router W_phi Conv1d kernel size (odd).",
    )
    p.add_argument(
        "--router_out_channels",
        type=int,
        default=128,
        help="When backbone=cnn: router W_phi Conv1d out channels; score scale uses √(L·C_out).",
    )
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="0 = infer from dataset (conditional); >0 = that many classes. --no_labels forces unconditional.",
    )
    p.add_argument(
        "--no_labels",
        action="store_true",
        help="Unconditional: no class embedding or aux head; ignore labels even if the dataset has them.",
    )
    p.add_argument(
        "--aux_beta",
        type=float,
        default=0.0,
        help="Weight for activity aux CE on DiT token hidden states (--backbone dit only; CNN matches SLM, no aux head).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device: "auto" (pick CUDA if usable, else CPU), "cpu", or "cuda"',
    )
    p.add_argument("--save", type=str, default="checkpoints/model.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", dest="use_wandb", action="store_true", default=True, help="Log to Weights & Biases")
    p.add_argument("--no-wandb", dest="use_wandb", action="store_false", help="Disable W&B")
    p.add_argument("--wandb_project", type=str, default="nonmarkovian", help="W&B project name")
    p.add_argument("--wandb_run_name", type=str, default="", help="Optional W&B run name")
    p.add_argument("--val_batch_size", type=int, default=0, help="Val batch size (0 = use --batch_size)")
    p.add_argument("--val_gen_batch", type=int, default=8, help="Batch size when generating sequences for FBD")
    p.add_argument(
        "--val_epoch_freq",
        type=int,
        default=1,
        help="Run validation loss (validate_routed) every N epochs (default 1 = every epoch).",
    )
    p.add_argument(
        "--fbd_epoch_freq",
        type=int,
        default=10,
        help="Run FBD metric + DNA preview every N epochs (default 5; set 1 for every epoch).",
    )
    p.add_argument(
        "--fbcnn_ckpt",
        type=str,
        default="",
        help="Path to fly-brain CNN checkpoint (FBCNN.ckpt) for FBD embeddings; empty = use denoiser encoder",
    )
    p.add_argument("--fbcnn_num_cls", type=int, default=81, help="Classifier num classes (fly brain: 81)")
    p.add_argument(
        "--fbcnn_stacks",
        type=int,
        default=1,
        help="CNNModel stacks (5 conv layers per stack). Fly-brain FBCNN.ckpt has 1 stack; "
        "using 4 with that ckpt leaves most weights random and FBD near0.",
    )
    p.add_argument(
        "--log_timing",
        action="store_true",
        help="Log per-batch wall times (ms, CUDA-synced): views, forward, loss, backward",
    )
    args, unknown = p.parse_known_args()

    dit_p = argparse.ArgumentParser(add_help=False)
    dit_p.add_argument("--nhead", type=int, default=8)
    dit_p.add_argument("--dec_layers", type=int, default=6, help="Number of DDiT blocks (single stack)")
    dit_p.add_argument("--dim_ff", type=int, default=1024)
    dit_p.add_argument("--cond_dim", type=int, default=0, help="AdaLN conditioning dim (0 = same as d_model)")
    dit_p.add_argument(
        "--time_freq_dim",
        type=int,
        default=256,
        help="Sinusoidal timestep embedding dim before MLP (baseline often uses 128)",
    )
    dit_p.add_argument("--dropout", type=float, default=0.1)

    if args.backbone == "dit":
        d_extra, tail = dit_p.parse_known_args(unknown)
        if tail:
            p.error("unrecognized arguments: " + " ".join(tail))
        args = argparse.Namespace(**{**vars(args), **vars(d_extra)})
    elif unknown:
        p.error(
            "unrecognized arguments (with --backbone cnn, omit DiT-only flags "
            "--nhead, --dec_layers, --dim_ff, --dropout, --cond_dim, --time_freq_dim): "
            + " ".join(unknown)
        )

    return args


def main() -> None:
    args = _parse_train_args()
    if args.num_timesteps_sample <= 0:
        args.num_timesteps_sample = args.num_timesteps

    ddp, rank, world_size, local_rank = setup_process_group()
    if ddp and not torch.cuda.is_available():
        cleanup_process_group()
        raise SystemExit("Multi-GPU training uses NCCL and requires CUDA.")
    if ddp:
        atexit.register(cleanup_process_group)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = resolve_device_arg(args.device)

    torch.manual_seed(args.seed)

    use_wandb = bool(args.use_wandb and wandb is not None and rank == 0)
    if rank == 0 and args.use_wandb and wandb is None:
        print("wandb not installed; pip install wandb. Continuing without W&B logging.")

    try:
        dfm_root_resolved = resolve_dfm_enhancer_root(args.dfm_enhancer, melanoma=args.dfm_melanoma)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    if not dfm_root_resolved:
        raise SystemExit("Could not resolve --dfm_enhancer (try auto with data_dfm/ from Zenodo 10184648).")

    args.dfm_enhancer = dfm_root_resolved
    train_ds_dfm = DFMEnhancerDataset(
        dfm_root_resolved,
        "train",
        melanoma=args.dfm_melanoma,
        max_len=args.max_len,
    )
    if args.no_labels:
        args.num_classes = 0
    elif args.num_classes <= 0:
        args.num_classes = train_ds_dfm.num_classes
    ds = train_ds_dfm

    def collate(b):
        return collate_pad(b)

    val_ds_dfm = DFMEnhancerDataset(
        dfm_root_resolved,
        "val",
        melanoma=args.dfm_melanoma,
        max_len=args.max_len,
    )
    test_ds_dfm = DFMEnhancerDataset(
        dfm_root_resolved,
        "test",
        melanoma=args.dfm_melanoma,
        max_len=args.max_len,
    )
    train_sampler: DistributedSampler | None = None
    if ddp:
        train_sampler = DistributedSampler(train_ds_dfm, shuffle=True)
    loader = DataLoader(
        train_ds_dfm,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=0,
    )
    vb = args.val_batch_size if args.val_batch_size > 0 else args.batch_size
    val_sampler: DistributedSampler | None = None
    if ddp:
        val_sampler = DistributedSampler(val_ds_dfm, shuffle=False, drop_last=False)
    val_loader = DataLoader(
        val_ds_dfm,
        batch_size=vb,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate,
        num_workers=0,
    )
    test_loader = DataLoader(test_ds_dfm, batch_size=vb, shuffle=False, collate_fn=collate, num_workers=0)

    num_labels = args.num_classes if args.num_classes > 0 else None
    if args.backbone == "cnn":
        model = RoutedDenoiserCNN(
            d_model=args.d_model,
            max_len=args.max_len,
            num_timesteps=args.num_timesteps,
            num_labels=num_labels,
            router_tau=args.router_tau,
            router_k=args.router_k,
            num_cnn_stacks=args.cnn_stacks,
            router_conv_kernel=args.router_conv_kernel,
            router_out_channels=args.router_out_channels,
        ).to(device)
    else:
        cond_dim = args.cond_dim if args.cond_dim > 0 else None
        model = RoutedDenoiser(
            d_model=args.d_model,
            nhead=args.nhead,
            dec_layers=args.dec_layers,
            dim_ff=args.dim_ff,
            dropout=args.dropout,
            max_len=args.max_len,
            num_timesteps=args.num_timesteps,
            num_labels=num_labels,
            cond_dim=cond_dim,
            router_tau=args.router_tau,
            router_k=args.router_k,
            time_freq_dim=args.time_freq_dim,
        ).to(device)

    aux_head: ActivityAuxHead | None = None
    if args.backbone == "dit" and args.aux_beta > 0 and num_labels is not None:
        aux_head = ActivityAuxHead(args.d_model, num_labels).to(device)

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        # loss_bal may be unused when --router_lambda_bal 0
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if aux_head is not None:
            aux_head = DDP(aux_head, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    n_params = sum(p.numel() for p in unwrap_ddp(model).parameters())
    n_trainable_model = sum(p.numel() for p in unwrap_ddp(model).parameters() if p.requires_grad)
    n_aux = sum(p.numel() for p in unwrap_ddp(aux_head).parameters()) if aux_head else 0
    n_trainable_aux = (
        sum(p.numel() for p in unwrap_ddp(aux_head).parameters() if p.requires_grad) if aux_head else 0
    )

    if rank == 0:
        print(
            f"model parameters: total={n_params:,} ({n_params / 1e6:.3f}M)  "
            f"trainable={n_trainable_model:,} ({n_trainable_model / 1e6:.3f}M)"
        )
        if aux_head:
            print(
                f"aux_head parameters: total={n_aux:,}  trainable={n_trainable_aux:,}  "
                f"combined trainable={n_trainable_model + n_trainable_aux:,} ({(n_trainable_model + n_trainable_aux) / 1e6:.3f}M)"
            )
        if ddp:
            print(
                f"Distributed: world_size={world_size}  per-GPU batch={args.batch_size}  "
                f"global_batch={args.batch_size * world_size}"
            )

    fbcnn = None
    if rank == 0 and args.fbcnn_ckpt.strip():
        from nonmarkovian.fbcnn import load_fbcnn_classifier

        fbcnn = load_fbcnn_classifier(
            args.fbcnn_ckpt,
            device,
            num_cls=args.fbcnn_num_cls,
            num_cnn_stacks=args.fbcnn_stacks,
        )

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config=vars(args),
        )
        wandb.summary["model/total_parameters"] = n_params
        wandb.summary["model/trainable_model_parameters"] = n_trainable_model
        wandb.summary["model/trainable_parameters"] = n_trainable_model + n_trainable_aux
        wandb.log(
            {
                "model/total_parameters": float(n_params),
                "model/trainable_model_parameters": float(n_trainable_model),
                "model/trainable_parameters": float(n_trainable_model + n_trainable_aux),
            },
            step=0,
        )
        if aux_head:
            wandb.summary["model/aux_head_parameters"] = n_aux
            wandb.summary["model/trainable_aux_parameters"] = n_trainable_aux
        wandb.summary["device"] = str(device)
        wandb.summary["cuda_usable"] = cuda_is_usable()
        if device.type == "cuda":
            wandb.summary["cuda_device_name"] = torch.cuda.get_device_name(device)
        wandb.summary["distributed"] = ddp
        wandb.summary["world_size"] = world_size
        if fbcnn is not None:
            wandb.summary["fbd_embedding"] = "fbcnn"
            wandb.summary["fbcnn_ckpt"] = str(Path(args.fbcnn_ckpt).resolve())

    try:
        _train_loop(
            args,
            device,
            ds,
            loader,
            val_loader,
            test_loader,
            model,
            aux_head,
            fbcnn,
            use_wandb,
            rank=rank,
            ddp=ddp,
            train_sampler=train_sampler,
        )
    finally:
        if use_wandb:
            wandb.finish()


def _train_loop(
    args: argparse.Namespace,
    device: torch.device,
    ds,
    loader: DataLoader,
    val_loader: DataLoader | None,
    test_loader: DataLoader | None,
    model: torch.nn.Module,
    aux_head: ActivityAuxHead | torch.nn.Module | None,
    fbcnn,
    use_wandb: bool,
    *,
    rank: int,
    ddp: bool,
    train_sampler: DistributedSampler | None,
) -> None:
    if use_wandb:
        wandb.summary["dataset_size"] = len(ds)
        if val_loader is not None:
            wandb.summary["val_dataset_size"] = len(val_loader.dataset)
        if test_loader is not None:
            wandb.summary["test_dataset_size"] = len(test_loader.dataset)

    alphas = cosine_alpha_schedule(args.num_timesteps, device=device)
    alphas_sample = cosine_alpha_schedule(args.num_timesteps_sample, device=device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + (list(aux_head.parameters()) if aux_head else []),
        lr=args.lr,
        weight_decay=0.01,
    )
    ema: _EMA | None = None
    if float(getattr(args, "ema_decay", 0.0)) > 0.0:
        ema_params = list(unwrap_ddp(model).parameters())
        if aux_head is not None:
            ema_params += list(unwrap_ddp(aux_head).parameters())
        ema = _EMA(ema_params, decay=float(args.ema_decay))

    if rank == 0:
        if val_loader is not None:
            print_val_and_random_dna_preview(
                val_loader.dataset,
                max_len=args.max_len,
                base_seed=args.seed,
                n=4,
            )
        aux_for_chance = args.aux_beta if args.backbone == "dit" else 0.0
        chance = chance_validation_baselines(
            aux_beta=aux_for_chance,
            num_classes=args.num_classes,
        )
        diff_scale = 1.0 if args.without_T else float(args.num_timesteps)
        chance["val/chance_baseline_diff"] = float(chance["val/chance_baseline_diff"] * diff_scale)
        aux_uniform = 0.0
        if args.backbone == "dit" and args.aux_beta > 0.0 and args.num_classes > 1:
            aux_uniform = float(args.aux_beta * math.log(float(args.num_classes)))
        chance["val/chance_baseline_loss"] = float(chance["val/chance_baseline_diff"] + aux_uniform)
        extras: list[str] = []
        if args.backbone == "dit" and args.aux_beta > 0 and args.num_classes > 1:
            extras.append("(val/loss baseline includes uniform aux)")
        if test_loader is not None:
            n_fbd0 = len(test_loader.dataset)
            if n_fbd0 >= 2:
                fbd_rand = compute_fbd_uniform_random_baseline(
                    unwrap_ddp(model),
                    test_loader,
                    device,
                    args,
                    n_samples=n_fbd0,
                    seq_len=args.max_len,
                    fbcnn=fbcnn,
                )
                chance["test/fbd_random_dna_baseline"] = float(fbd_rand)
                extras.append(f"FBD random-DNA vs test (n={n_fbd0}) ~ {fbd_rand:.4f}")
            else:
                extras.append("pre-train FBD skipped (test split has <2 examples)")
        suffix = ("  " + "  ".join(extras)) if extras else ""
        print(
            f"Chance baseline (uniform 4-class logits, mean NLL = log 4, scale={diff_scale:g}):  "
            f"val/diff_loss ~ {chance['val/chance_baseline_diff']:.4f}  "
            f"val/loss ~ {chance['val/chance_baseline_loss']:.4f}{suffix}"
        )
        if use_wandb:
            wandb.log(chance, step=0)

    best_val_loss = float("inf")
    best_save_path: Path | None = None
    global_step = 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        if aux_head:
            aux_head.train()
        total_loss = 0.0
        n_batches = 0
        num_batches = len(loader)
        sum_ms_views = sum_ms_fwd = sum_ms_loss = sum_ms_bwd = 0.0
        for batch_idx, batch in enumerate(loader):
            x0 = batch["x0"].to(device)
            pad = batch["mask_pad"].to(device)
            if _use_conditional_sampling_labels(args):
                labels = batch.get("label")
                if labels is not None:
                    labels = labels.to(device)
            else:
                labels = None

            B, L = x0.shape
            gen = torch.Generator(device=x0.device)
            gen.manual_seed(global_step + epoch * 10000)
            if args.log_timing:
                t0 = tic(device)
            views = sample_all_views_bernoulli(
                x0,
                args.num_timesteps,
                scheduler=args.bernoulli_scheduler,
                generator=gen,
            )
            ms_views = toc_ms(t0, device) if args.log_timing else 0.0

            t_start = int(torch.randint(0, args.num_timesteps, (1,), device=device).item())
            t_cont = float(t_start + 1) / float(args.num_timesteps)
            if args.log_timing:
                t0 = tic(device)
            labels_in = labels
            if labels is not None and args.cond_drop_prob > 0:
                keep = torch.rand((B,), device=device) >= float(args.cond_drop_prob)
                if args.backbone == "cnn":
                    null_cls = int(args.num_classes)
                    labels_in = torch.where(keep, labels, torch.full_like(labels, null_cls))
                elif not bool(keep.all()):
                    labels_in = None
            logits, pi, h_dec, loss_bal, _seq_in = model(views, t_start, labels=labels_in, t_cond=t_cont)
            aux_loss_val: torch.Tensor | None = None
            if aux_head is not None and labels is not None and args.aux_beta > 0:
                aux_logits = aux_head(h_dec)
                aux_loss_val = F.cross_entropy(aux_logits, labels)
            ms_fwd = toc_ms(t0, device) if args.log_timing else 0.0

            if args.log_timing:
                t0 = tic(device)
            target = x0.clamp(max=3)
            log_probs = F.log_softmax(logits, dim=-1)
            nlog_p = -torch.gather(log_probs, -1, target[:, :, None]).squeeze(-1)
            if not args.without_T:
                nlog_p = float(args.num_timesteps) * nlog_p
            nlog_p = nlog_p.masked_fill(pad, 0.0)
            denom = (~pad).float().sum().clamp(min=1.0)
            diff_loss = nlog_p.float().sum() / denom

            loss = diff_loss
            if args.router_lambda_bal > 0:
                loss = loss + args.router_lambda_bal * loss_bal
            if aux_loss_val is not None:
                loss = loss + args.aux_beta * aux_loss_val
            ms_loss = toc_ms(t0, device) if args.log_timing else 0.0

            if args.log_timing:
                t0 = tic(device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm_model = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm_aux: torch.Tensor | None = None
            if aux_head:
                grad_norm_aux = torch.nn.utils.clip_grad_norm_(aux_head.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update()
            ms_bwd = toc_ms(t0, device) if args.log_timing else 0.0

            if args.log_timing:
                sum_ms_views += ms_views
                sum_ms_fwd += ms_fwd
                sum_ms_loss += ms_loss
                sum_ms_bwd += ms_bwd

            total_loss += float(loss.item())
            n_batches += 1
            global_step += 1

            if use_wandb:
                with torch.no_grad():
                    if pi.shape[-1] > 0:
                        p = pi.clamp(min=1e-8)
                        ent = -(p * p.log()).sum(dim=-1).mean()
                    else:
                        ent = torch.tensor(0.0, device=x0.device)
                num_tokens = int((~pad).sum().item())
                log_payload: dict = {
                    "train/loss": float(loss.item()),
                    "train/diff_loss": float(diff_loss.item()),
                    "train/t_start": t_start,
                    "train/learning_rate": opt.param_groups[0]["lr"],
                    "train/grad_norm_model": _to_float(grad_norm_model),
                    "train/router_entropy": float(ent.item()),
                    "train/router_num_candidates": int(pi.shape[-1]),
                    "train/batch_idx": batch_idx,
                    "train/batches_per_epoch": num_batches,
                    "train/batch_size": B,
                    "train/seq_len_padded": L,
                    "train/num_tokens": num_tokens,
                    "epoch": epoch + 1,
                }
                if pi.shape[-1] > 0:
                    with torch.no_grad():
                        am = pi.argmax(dim=-1)
                        log_payload["train/router_argmax_mean"] = float(am.float().mean())
                        if B > 1:
                            log_payload["train/router_argmax_std"] = float(am.float().std(unbiased=False))
                        log_payload["train/router_argmax_min"] = int(am.min().item())
                        log_payload["train/router_argmax_max"] = int(am.max().item())
                        if pi.shape[-1] > 1:
                            log_payload["train/router_weight_last_mean"] = float(pi[:, -1].mean())
                        else:
                            log_payload["train/router_weight_last_mean"] = 1.0
                        last_idx = pi.shape[-1] - 1
                        last_chosen = int((am == last_idx).sum().item())
                        log_payload["train/router_last_chosen_count"] = last_chosen
                        log_payload["train/router_last_chosen_frac"] = last_chosen / float(B)
                        log_payload["train/router_argmax_hist"] = wandb.Histogram(am.detach().cpu().numpy())
                if args.router_lambda_bal > 0:
                    log_payload["train/router_loss_balance"] = float(loss_bal.item())
                if aux_loss_val is not None:
                    log_payload["train/aux_loss"] = float(aux_loss_val.item())
                if grad_norm_aux is not None:
                    log_payload["train/grad_norm_aux"] = _to_float(grad_norm_aux)
                log_payload["train/router_tau"] = args.router_tau
                log_payload["train/router_k"] = args.router_k
                if args.log_timing:
                    log_payload["train/time_ms_views"] = ms_views
                    log_payload["train/time_ms_forward"] = ms_fwd
                    log_payload["train/time_ms_loss"] = ms_loss
                    log_payload["train/time_ms_backward"] = ms_bwd
                wandb.log(log_payload, step=global_step)

        avg = total_loss / max(n_batches, 1)
        if rank == 0:
            print(f"epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}")
        if args.log_timing and n_batches > 0 and rank == 0:
            print(
                f"  timing_ms (batch avg): sample_all_views={sum_ms_views / n_batches:.2f}  "
                f"forward={sum_ms_fwd / n_batches:.2f}  loss={sum_ms_loss / n_batches:.2f}  "
                f"backward+step={sum_ms_bwd / n_batches:.2f}"
            )
            if use_wandb:
                wandb.log(
                    {
                        "train/epoch_time_ms_views_avg": sum_ms_views / n_batches,
                        "train/epoch_time_ms_forward_avg": sum_ms_fwd / n_batches,
                        "train/epoch_time_ms_loss_avg": sum_ms_loss / n_batches,
                        "train/epoch_time_ms_backward_avg": sum_ms_bwd / n_batches,
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )
        if use_wandb:
            wandb.log({"train/epoch_loss_avg": avg, "epoch": epoch + 1}, step=global_step)

        if val_loader is not None:
            m = unwrap_ddp(model)
            val_ds = val_loader.dataset
            ah = unwrap_ddp(aux_head) if aux_head is not None else None
            if ema is not None:
                ema.store()
                ema.copy_to()
            do_val_loss = (epoch + 1) % args.val_epoch_freq == 0
            do_fbd = (epoch + 1) % args.fbd_epoch_freq == 0

            if do_val_loss:
                vmetrics = validate_routed(
                    m,
                    val_loader,
                    device,
                    ah,
                    args,
                    epoch=epoch,
                    global_step=global_step,
                )
                if rank == 0:
                    val_noh = vmetrics.get("val/loss_no_history")
                    if val_noh is not None:
                        print(
                            f"  val_loss={vmetrics['val/loss']:.4f}  "
                            f"val_loss_no_history={float(val_noh):.4f}"
                        )
                    else:
                        print(f"  val_loss={vmetrics['val/loss']:.4f}")
                    if use_wandb:
                        wandb.log(vmetrics, step=global_step)
                    # Best checkpoint tracks the with-history val/loss (training distribution).
                    cur_val = float(vmetrics["val/loss"])
                    if cur_val < best_val_loss:
                        best_val_loss = cur_val
                        save_path = _resolve_save_path(args.save, use_wandb)
                        best_save_path = save_path.with_name(f"{save_path.stem}.best{save_path.suffix}")
                        best_save_path.parent.mkdir(parents=True, exist_ok=True)
                        best_payload = {
                            "model": m.state_dict(),
                            "args": vars(args),
                            "alphas": alphas.cpu(),
                            "alphas_sample": alphas_sample.cpu(),
                            "trainer": "routed_discrete",
                            "best_val_loss": best_val_loss,
                            "best_val_loss_no_history": (
                                float(val_noh) if val_noh is not None else None
                            ),
                            "best_epoch": epoch + 1,
                            "best_global_step": global_step,
                        }
                        if ah is not None:
                            best_payload["aux_head"] = ah.state_dict()
                        torch.save(best_payload, best_save_path)
                        print(f"  best checkpoint updated: {best_save_path} (val/loss={best_val_loss:.4f})")
                        if use_wandb:
                            wandb.summary["checkpoint_best_path"] = str(best_save_path.resolve())
                            wandb.summary["checkpoint_best_val_loss"] = best_val_loss
                            if val_noh is not None:
                                wandb.summary["checkpoint_best_val_loss_no_history"] = float(val_noh)

            if do_fbd:
                n_fbd = len(val_loader.dataset)
                if n_fbd >= 2:
                    fbd = compute_fbd_routed(
                        m,
                        val_loader,
                        alphas_sample,
                        device,
                        args,
                        n_samples=n_fbd,
                        seq_len=args.max_len,
                        epoch=epoch,
                        fbcnn=fbcnn,
                    )
                    if rank == 0:
                        tag = "fbd_fbcnn" if fbcnn is not None else "fbd"
                        print(f"  {tag}={fbd:.4f}")
                        if use_wandb:
                            wandb.log({"val/fbd": float(fbd), "epoch": epoch + 1}, step=global_step)
                elif rank == 0:
                    print("  fbd=skipped (<2 val examples)")
                print_epoch_diffusion_dna_samples(
                    m,
                    alphas_sample,
                    device,
                    args,
                    val_ds,
                    epoch=epoch,
                    global_step=global_step,
                    n=4,
                    routed=True,
                )
            if ema is not None:
                ema.restore()
        if ddp:
            barrier()

    if rank == 0:
        save_path = _resolve_save_path(args.save, use_wandb)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": unwrap_ddp(model).state_dict(),
            "args": vars(args),
            "alphas": alphas.cpu(),
            "alphas_sample": alphas_sample.cpu(),
            "trainer": "routed_discrete",
        }
        if aux_head is not None:
            payload["aux_head"] = unwrap_ddp(aux_head).state_dict()
        torch.save(payload, save_path)
        print(f"saved {save_path}")
        if best_save_path is not None:
            print(f"best val checkpoint: {best_save_path} (val/loss={best_val_loss:.4f})")

        if use_wandb:
            wandb.summary["checkpoint_path"] = str(save_path.resolve())
    if ddp:
        barrier()


if __name__ == "__main__":
    main()
