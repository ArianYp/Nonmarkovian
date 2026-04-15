"""Train non-Markovian routed discrete diffusion on DFM enhancer pickles (Zenodo)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nonmarkovian.data import DFMEnhancerDataset, collate_pad, resolve_dfm_enhancer_root
from nonmarkovian.device_utils import cuda_is_usable, resolve_device_arg
from nonmarkovian.forward import cosine_alpha_schedule, sample_all_views
from nonmarkovian.model import ActivityAuxHead, RoutedDenoiser
from nonmarkovian.train_timing import tic, toc_ms
from nonmarkovian.validation import compute_fbd_routed, validate_routed

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[misc, assignment]


def _to_float(x: torch.Tensor | float) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    return float(x)


def timestep_loss_weight(alphas: torch.Tensor, t_start: int) -> float:
    """λ_t from schedule increment (paper: corruption-based weighting)."""
    if t_start == 0:
        return float(alphas[0].item())
    return float((alphas[t_start] - alphas[t_start - 1]).clamp(min=1e-6).item())


def main() -> None:
    p = argparse.ArgumentParser()
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
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_timesteps", type=int, default=32)
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
        default=0.0,
        help="Weight λ for load-balancing loss (Switch-style; 0 = off)",
    )
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dec_layers", type=int, default=6, help="Number of DDiT blocks (single stack)")
    p.add_argument("--dim_ff", type=int, default=1024)
    p.add_argument("--cond_dim", type=int, default=0, help="AdaLN conditioning dim (0 = same as d_model)")
    p.add_argument(
        "--time_freq_dim",
        type=int,
        default=256,
        help="Sinusoidal timestep embedding dim before MLP (baseline often uses 128)",
    )
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_classes", type=int, default=0, help=">0 enables label embedding + optional aux head")
    p.add_argument("--aux_beta", type=float, default=0.0, help="Weight for activity aux CE (needs labels)")
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
    p.add_argument(
        "--val_fbd_n",
        type=int,
        default=0,
        help="Per-epoch FBD: number of real/generated sequence *pairs* to compare (0 = skip). "
        "Not sequence length; generation length is --max_len.",
    )
    p.add_argument("--val_gen_batch", type=int, default=8, help="Batch size when generating sequences for FBD")
    p.add_argument(
        "--fbcnn_ckpt",
        type=str,
        default="",
        help="Path to fly-brain CNN checkpoint (FBCNN.ckpt) for FBD embeddings; empty = use denoiser encoder",
    )
    p.add_argument("--fbcnn_num_cls", type=int, default=81, help="Classifier num classes (fly brain: 81)")
    p.add_argument("--fbcnn_stacks", type=int, default=4, help="num_cnn_stacks for CNNModel(4, num_cls, stacks)")
    p.add_argument(
        "--log_timing",
        action="store_true",
        help="Log per-batch wall times (ms, CUDA-synced): views, forward, loss, backward",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device_arg(args.device)

    use_wandb = bool(args.use_wandb and wandb is not None)
    if args.use_wandb and wandb is None:
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
    if args.num_classes <= 0:
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
    loader = DataLoader(
        train_ds_dfm, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    vb = args.val_batch_size if args.val_batch_size > 0 else args.batch_size
    val_loader = DataLoader(val_ds_dfm, batch_size=vb, shuffle=False, collate_fn=collate, num_workers=0)

    num_labels = args.num_classes if args.num_classes > 0 else None
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
    if args.aux_beta > 0 and num_labels is not None:
        aux_head = ActivityAuxHead(args.d_model, num_labels).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_aux = sum(p.numel() for p in aux_head.parameters()) if aux_head else 0
    n_trainable_aux = sum(p.numel() for p in aux_head.parameters() if p.requires_grad) if aux_head else 0

    print(
        f"model parameters: total={n_params:,} ({n_params / 1e6:.3f}M)  "
        f"trainable={n_trainable_model:,} ({n_trainable_model / 1e6:.3f}M)"
    )
    if aux_head:
        print(
            f"aux_head parameters: total={n_aux:,}  trainable={n_trainable_aux:,}  "
            f"combined trainable={n_trainable_model + n_trainable_aux:,} ({(n_trainable_model + n_trainable_aux) / 1e6:.3f}M)"
        )

    fbcnn = None
    if args.fbcnn_ckpt.strip():
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
        if fbcnn is not None:
            wandb.summary["fbd_embedding"] = "fbcnn"
            wandb.summary["fbcnn_ckpt"] = str(Path(args.fbcnn_ckpt).resolve())

    try:
        _train_loop(args, device, ds, loader, val_loader, model, aux_head, fbcnn, use_wandb)
    finally:
        if use_wandb:
            wandb.finish()


def _train_loop(
    args: argparse.Namespace,
    device: torch.device,
    ds,
    loader: DataLoader,
    val_loader: DataLoader | None,
    model: RoutedDenoiser,
    aux_head: ActivityAuxHead | None,
    fbcnn,
    use_wandb: bool,
) -> None:
    if use_wandb:
        wandb.summary["dataset_size"] = len(ds)
        if val_loader is not None:
            wandb.summary["val_dataset_size"] = len(val_loader.dataset)

    alphas = cosine_alpha_schedule(args.num_timesteps, device=device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + (list(aux_head.parameters()) if aux_head else []),
        lr=args.lr,
        weight_decay=0.01,
    )

    global_step = 0
    for epoch in range(args.epochs):
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
            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(device)

            B, L = x0.shape
            gen = torch.Generator(device=x0.device)
            gen.manual_seed(global_step + epoch * 10000)
            if args.log_timing:
                t0 = tic(device)
            views = sample_all_views(x0, alphas, generator=gen)
            ms_views = toc_ms(t0, device) if args.log_timing else 0.0

            t_start = int(torch.randint(0, args.num_timesteps, (1,), device=device).item())
            if args.log_timing:
                t0 = tic(device)
            logits, pi, h_dec, loss_bal = model(views, t_start, labels=labels)
            aux_loss_val: torch.Tensor | None = None
            if aux_head is not None and labels is not None and args.aux_beta > 0:
                aux_logits = aux_head(h_dec)
                aux_loss_val = F.cross_entropy(aux_logits, labels)
            ms_fwd = toc_ms(t0, device) if args.log_timing else 0.0

            if args.log_timing:
                t0 = tic(device)
            target = x0.clamp(max=3)
            ce = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
            ce = ce.masked_fill(pad, 0.0)
            denom = (~pad).float().sum().clamp(min=1.0)
            w = timestep_loss_weight(alphas, t_start)
            diff_loss = ce.float().sum() / denom * w

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
                    "train/timestep_weight": w,
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
        print(f"epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}")
        if args.log_timing and n_batches > 0:
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
            vmetrics = validate_routed(
                model,
                val_loader,
                alphas,
                device,
                aux_head,
                args,
                epoch=epoch,
                global_step=global_step,
            )
            msg = f"  val_loss={vmetrics['val/loss']:.4f}"
            if args.val_fbd_n > 0:
                n_fbd = min(args.val_fbd_n, len(val_loader.dataset))
                seq_len_fbd = (
                    args.max_len
                )
                if n_fbd >= 2:
                    fbd = compute_fbd_routed(
                        model,
                        val_loader,
                        alphas,
                        device,
                        args,
                        n_samples=n_fbd,
                        seq_len=seq_len_fbd,
                        epoch=epoch,
                        fbcnn=fbcnn,
                    )
                    vmetrics["val/fbd"] = fbd
                    tag = "fbd_fbcnn" if fbcnn is not None else "fbd"
                    msg += f"  {tag}={fbd:.4f}"
                else:
                    print("  val_fbd_n too small or val set too small; skipping FBD")
            print(msg)
            if use_wandb:
                wandb.log(vmetrics, step=global_step)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "args": vars(args),
        "alphas": alphas.cpu(),
    }
    if aux_head:
        payload["aux_head"] = aux_head.state_dict()
    torch.save(payload, save_path)
    print(f"saved {save_path}")

    if use_wandb:
        wandb.summary["checkpoint_path"] = str(save_path.resolve())


if __name__ == "__main__":
    main()
