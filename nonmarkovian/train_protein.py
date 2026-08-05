"""Train the Non-Markovian routed diffusion on UniRef50 proteins with the SLM ``dit_bfn`` backbone.

This mirrors ``nonmarkovian.train`` (same routed multi-view machinery) but for the
protein task, and is byte-aligned with SLM's UniRef ``new_diff`` setup
(``makefile:train_uniref50``: ``model=evodiff``, ``backbone=dit_bfn``,
``parameterization=new_diff``, ``length=1024``, ``T=500``, ``beta`` n/a for new_diff):

- corruption: SLM ``get_xt_bernoulli`` Bernoulli simplex over ``vocab_size=31``
  (``slm_denominator=True``), but sampled as a *multi-view* tensor so the Boltzmann
  router can condition on noisier (history) views -- this is the non-Markovian part.
- denoiser: ``RoutedDenoiserDiTBFN`` (router + ``BFN_DIT`` with the ``evodiff`` config).
- loss: SLM ``new_diff`` *training* objective = ``T * NLL(x0)``, pad-masked by the
  attention mask (``slm.py:_new_diffusion_loss`` training branch + ``_loss`` masking).

Note on memory: the multi-view tensor is ``[B, T-t_start, L, V]``. With UniRef's
``L=1024``, ``V=31`` and large ``T`` this is sizeable -- lower ``--batch_size`` /
``--num_timesteps`` (or ``--max_len``) if you OOM.
"""

from __future__ import annotations

import argparse
import atexit
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nonmarkovian.data_protein import (
    UniRefDataset,
    WrappedUniRefDataset,
    collate_protein,
    resolve_uniref_root,
)
from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.distributed_utils import (
    barrier,
    cleanup_process_group,
    setup_process_group,
    unwrap_ddp,
)
from nonmarkovian.forward import sample_all_views_bernoulli
from nonmarkovian.model_protein import RoutedDenoiserDiTBFN
from nonmarkovian.vocab_protein import PAD_IDX, VOCAB_SIZE

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


class _EMA:
    def __init__(self, params, decay: float):
        self.params = [p for p in params if p.requires_grad]
        self.decay = float(decay)
        self.shadow = [p.detach().clone() for p in self.params]
        self.backup = None

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self):
        self.backup = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def copy_to(self):
        for p, s in zip(self.params, self.shadow):
            p.data.copy_(s)

    @torch.no_grad()
    def restore(self):
        if self.backup is None:
            return
        for p, b in zip(self.params, self.backup):
            p.data.copy_(b)
        self.backup = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    p.add_argument("--uniref", type=str, default="auto", help="UniRef dir with consensus.fasta (or 'auto').")
    p.add_argument("--max_len", type=int, default=1024, help="Sequence length (SLM uniref uses 1024).")
    p.add_argument(
        "--batch_size", type=int, default=32,
        help="Per-GPU batch (SLM uses 32/gpu x 4 gpus = global 128). Global = this * world_size.",
    )
    p.add_argument("--val_batch_size", type=int, default=0, help="0 = same as --batch_size")
    p.add_argument("--num_workers", type=int, default=4)
    # optim
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument(
        "--max_steps", type=int, default=200_000,
        help="Stop after this many optimizer steps (SLM/paper uses 200k; 0 = no cap).",
    )
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--ema_decay", type=float, default=0.9999, help="0 disables EMA.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    # diffusion / corruption (SLM new_diff)
    p.add_argument("--num_timesteps", type=int, default=500, help="Diffusion T (SLM uniref uses 500).")
    p.add_argument(
        "--bernoulli_scheduler", type=str, default="loglinear", choices=("loglinear", "linear"),
        help="SLM training.bscheduler (uniref default loglinear).",
    )
    p.add_argument(
        "--corruption_mode", type=str, default="independent", choices=("independent", "trajectory"),
        help="Multi-view corruption: 'independent' = fresh per-step noise (SLM new_diff style).",
    )
    p.add_argument("--without_T", action="store_true", help="Do not scale NLL by T (SLM training.without_T).")
    # router
    p.add_argument("--router_tau", type=float, default=0.01)
    p.add_argument("--router_k", type=int, default=1)
    p.add_argument("--router_lambda_bal", type=float, default=0.0)
    p.add_argument("--router_conv_kernel", type=int, default=1)
    p.add_argument("--router_out_channels", type=int, default=128)
    # model (SLM configs/model/evodiff.yaml defaults)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--cond_dim", type=int, default=128)
    p.add_argument("--n_blocks", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_scale_by_sigma", dest="scale_by_sigma", action="store_false", default=True)
    p.add_argument("--embedding_nml", action="store_true", help="SLM embedding_nml (default False).")
    p.add_argument("--entropy_condition", action="store_true", help="SLM entropy_condition (default False).")
    # misc
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="checkpoints/protein_routed.pt")
    p.add_argument("--val_epoch_freq", type=int, default=1)
    p.add_argument("--val_max_batches", type=int, default=50, help="Cap val batches per eval (0 = all).")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--wandb", dest="use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="nonmarkovian-protein")
    p.add_argument("--wandb_run_name", type=str, default="")
    return p.parse_args()


def _build_model(args, device) -> RoutedDenoiserDiTBFN:
    return RoutedDenoiserDiTBFN(
        vocab_size=VOCAB_SIZE,
        max_len=args.max_len,
        num_timesteps=args.num_timesteps,
        hidden_size=args.hidden_size,
        cond_dim=args.cond_dim,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        dropout=args.dropout,
        scale_by_sigma=args.scale_by_sigma,
        embedding_nml=args.embedding_nml,
        entropy_condition=args.entropy_condition,
        router_tau=args.router_tau,
        router_k=args.router_k,
        router_conv_kernel=args.router_conv_kernel,
        router_out_channels=args.router_out_channels,
    ).to(device)


def _diffusion_loss(model, x0, attn, args, device, gen):
    """One SLM-``new_diff`` training step on the routed multi-view.

    Returns ``(loss, diff_loss, loss_bal, pi, t_start)`` -- ``pi`` are the router
    weights ``[B, K]`` and ``t_start`` the sampled current timestep, both needed for
    the enhancer-style router logging.
    """
    B, L = x0.shape
    T = args.num_timesteps
    t_start = int(torch.randint(0, T, (1,), device=device).item())
    t_cont = float(t_start + 1) / float(T)
    views = sample_all_views_bernoulli(
        x0,
        T,
        t_start=t_start,
        scheduler=args.bernoulli_scheduler,
        generator=gen,
        num_classes=VOCAB_SIZE,
        corruption_mode=args.corruption_mode,
        slm_denominator=True,  # match SLM get_xt_bernoulli ((E[nums]-1)/vocab_size)
    )
    logits, pi, _h, loss_bal, _seq_in = model(
        views, 0, t_cond=t_cont, t_start_abs=t_start, scheduler=args.bernoulli_scheduler
    )
    log_probs = F.log_softmax(logits, dim=-1)
    nlog_p = -torch.gather(log_probs, -1, x0[:, :, None]).squeeze(-1)  # [B, L]
    if not args.without_T:
        nlog_p = float(T) * nlog_p
    # Pad-mask exactly like SLM (_loss: nlls = loss * attention_mask; mean over real tokens).
    nlls = nlog_p * attn
    denom = attn.sum().clamp(min=1.0)
    diff_loss = nlls.sum() / denom
    loss = diff_loss
    if args.router_lambda_bal > 0:
        loss = loss + args.router_lambda_bal * loss_bal
    return loss, diff_loss, loss_bal, pi, t_start


@torch.no_grad()
def _validate(model, val_loader, args, device) -> float:
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    total, nb = 0.0, 0
    for i, batch in enumerate(val_loader):
        if args.val_max_batches and i >= args.val_max_batches:
            break
        x0 = batch["x0"].to(device)
        attn = batch["attention_mask"].to(device)
        _l, diff_loss, *_ = _diffusion_loss(model, x0, attn, args, device, gen)
        total += float(diff_loss.item())
        nb += 1
    model.train()
    return total / max(nb, 1)


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    return float(x)


def _tag_save_path_with_run(save: str, use_wandb: bool) -> str:
    """Insert a unique run id into the checkpoint filename so runs don't overwrite each other.

    Uses the active W&B run id when available (so the checkpoint name matches the W&B run);
    otherwise generates one (``wandb.util.generate_id()`` or a uuid fallback).
    ``checkpoints/protein_routed.pt`` -> ``checkpoints/protein_routed_<runid>.pt``.
    """
    p = Path(save)
    run_id = None
    if use_wandb and wandb is not None and getattr(wandb, "run", None) is not None:
        run_id = wandb.run.id
    if not run_id:
        try:
            run_id = wandb.util.generate_id() if wandb is not None else None
        except Exception:
            run_id = None
    if not run_id:
        import uuid

        run_id = uuid.uuid4().hex[:8]
    return str(p.with_name(f"{p.stem}_{run_id}{p.suffix}"))


def _router_log_payload(
    loss, diff_loss, loss_bal, pi, t_start, grad_norm, opt, args,
    *, B, L, num_real_tokens, batch_idx, num_batches, epoch,
) -> dict:
    """W&B payload mirroring ``nonmarkovian.train`` (router entropy + argmax stats).

    Router candidate ``k`` corresponds to absolute timestep ``t_start + 1 + k`` (the
    views handed to the model start at the current step ``t_start``), so argmax stats
    are reported both as candidate indices and absolute timesteps, as in the enhancer.
    """
    with torch.no_grad():
        if pi.shape[-1] > 0:
            p = pi.clamp(min=1e-8)
            ent = float((-(p * p.log()).sum(dim=-1).mean()).item())
        else:
            ent = 0.0
    payload: dict = {
        "train/loss": float(loss.item()),
        "train/diff_loss": float(diff_loss.item()),
        "train/t_start": int(t_start),
        "train/learning_rate": opt.param_groups[0]["lr"],
        "train/grad_norm_model": _to_float(grad_norm),
        "train/router_entropy": ent,
        "train/router_num_candidates": int(pi.shape[-1]),
        "train/router_tau": args.router_tau,
        "train/router_k": args.router_k,
        "train/batch_idx": int(batch_idx),
        "train/batches_per_epoch": int(num_batches),
        "train/batch_size": int(B),
        "train/seq_len_padded": int(L),
        "train/num_tokens": int(num_real_tokens),
        "epoch": epoch + 1,
    }
    if pi.shape[-1] > 0:
        with torch.no_grad():
            am = pi.argmax(dim=-1)
            chosen_t_abs = am + (t_start + 1)
            delta = t_start - chosen_t_abs
            payload["train/router_argmax_mean"] = float(am.float().mean())
            if B > 1:
                payload["train/router_argmax_std"] = float(am.float().std(unbiased=False))
            payload["train/router_argmax_min"] = int(am.min().item())
            payload["train/router_argmax_max"] = int(am.max().item())
            payload["train/router_argmax_abs_t_mean"] = float(chosen_t_abs.float().mean())
            payload["train/router_argmax_abs_t_min"] = int(chosen_t_abs.min().item())
            payload["train/router_argmax_abs_t_max"] = int(chosen_t_abs.max().item())
            payload["train/router_t_start_minus_argmax_t_mean"] = float(delta.float().mean())
            payload["train/router_t_start_minus_argmax_t_min"] = int(delta.min().item())
            payload["train/router_t_start_minus_argmax_t_max"] = int(delta.max().item())
            if pi.shape[-1] > 1:
                payload["train/router_weight_last_mean"] = float(pi[:, -1].mean())
            else:
                payload["train/router_weight_last_mean"] = 1.0
            last_idx = pi.shape[-1] - 1
            last_chosen = int((am == last_idx).sum().item())
            payload["train/router_last_chosen_count"] = last_chosen
            payload["train/router_last_chosen_frac"] = last_chosen / float(B)
            payload["train/router_argmax_hist"] = wandb.Histogram(am.detach().cpu().numpy())
    if args.router_lambda_bal > 0:
        payload["train/router_loss_balance"] = float(loss_bal.item())
    return payload


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

    root = resolve_uniref_root(args.uniref)
    if rank == 0:
        print(f"UniRef root: {root}  vocab_size={VOCAB_SIZE}  T={args.num_timesteps}  L={args.max_len}")
    train_ds = WrappedUniRefDataset(UniRefDataset(root, "train", max_len=args.max_len), max_len=args.max_len)
    val_ds = WrappedUniRefDataset(UniRefDataset(root, "valid", max_len=args.max_len), max_len=args.max_len)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_protein,
        num_workers=args.num_workers,
        drop_last=True,
    )
    vb = args.val_batch_size if args.val_batch_size > 0 else args.batch_size
    val_loader = DataLoader(
        val_ds, batch_size=vb, shuffle=False, collate_fn=collate_protein, num_workers=args.num_workers
    )

    model = _build_model(args, device)
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    n_params = sum(p.numel() for p in unwrap_ddp(model).parameters())
    if rank == 0:
        print(
            f"model parameters: {n_params:,} ({n_params / 1e6:.2f}M)  "
            f"global_batch={args.batch_size * world_size} (={args.batch_size}/gpu x {world_size})"
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = None
    if float(args.ema_decay) > 0.0:
        ema = _EMA(list(unwrap_ddp(model).parameters()), decay=float(args.ema_decay))

    use_wandb = bool(args.use_wandb and wandb is not None and rank == 0)
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or None, config=vars(args))
        wandb.summary["model/total_parameters"] = n_params

    # Make checkpoints unique per run (W&B run id) so concurrent/repeat runs don't overwrite.
    if rank == 0:
        args.save = _tag_save_path_with_run(args.save, use_wandb)
        print(f"checkpoints -> {args.save}  (best: {Path(args.save).stem}.best{Path(args.save).suffix})")
        if use_wandb:
            wandb.summary["checkpoint_path"] = str(Path(args.save).resolve())

    best_val = float("inf")
    global_step = 0
    stop = False
    num_batches = len(train_loader)
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        for batch_idx, batch in enumerate(train_loader):
            x0 = batch["x0"].to(device)
            attn = batch["attention_mask"].to(device)
            B, L = x0.shape
            gen = torch.Generator(device=device)
            gen.manual_seed(global_step + epoch * 1_000_003)

            loss, diff_loss, loss_bal, pi, t_start = _diffusion_loss(model, x0, attn, args, device, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if ema is not None:
                ema.update()

            running += float(loss.item())
            global_step += 1
            if rank == 0 and args.log_every > 0 and global_step % args.log_every == 0:
                print(
                    f"epoch {epoch + 1} step {global_step} "
                    f"loss={loss.item():.4f} diff={diff_loss.item():.4f} gnorm={float(grad_norm):.2f}"
                )
            if use_wandb:
                log_payload = _router_log_payload(
                    loss, diff_loss, loss_bal, pi, t_start, grad_norm, opt,
                    args, B=B, L=L, num_real_tokens=int(attn.sum().item()),
                    batch_idx=batch_idx, num_batches=num_batches, epoch=epoch,
                )
                wandb.log(log_payload, step=global_step)
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break

        if rank == 0:
            print(f"epoch {epoch + 1}/{args.epochs} avg_loss={running / max(batch_idx + 1, 1):.4f}")

        # validation (EMA weights)
        if (epoch + 1) % args.val_epoch_freq == 0:
            m = unwrap_ddp(model)
            if ema is not None:
                ema.store()
                ema.copy_to()
            val_loss = _validate(m, val_loader, args, device)
            if ema is not None:
                ema.restore()
            if rank == 0:
                print(f"  val_diff_loss={val_loss:.4f}")
                if use_wandb:
                    wandb.log({"val/diff_loss": val_loss, "epoch": epoch + 1}, step=global_step)
                if val_loss < best_val:
                    best_val = val_loss
                    save_path = Path(args.save)
                    best_path = save_path.with_name(f"{save_path.stem}.best{save_path.suffix}")
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"model": m.state_dict(), "args": vars(args), "trainer": "protein_routed_ditbfn",
                         "best_val": best_val, "best_epoch": epoch + 1, "vocab_size": VOCAB_SIZE},
                        best_path,
                    )
                    print(f"  best checkpoint -> {best_path} (val_diff_loss={best_val:.4f})")
        if ddp:
            barrier()
        if stop:
            break

    if rank == 0:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": unwrap_ddp(model).state_dict(), "args": vars(args),
             "trainer": "protein_routed_ditbfn", "vocab_size": VOCAB_SIZE},
            save_path,
        )
        print(f"saved {save_path}")
        if use_wandb:
            wandb.finish()
    if ddp:
        barrier()


if __name__ == "__main__":
    main()
