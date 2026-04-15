"""Iterative sampling: synthetic multi-scale views from predicted x0 (inference-time non-Markovian views)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.forward import corrupt_sequence, cosine_alpha_schedule
from nonmarkovian.model import RoutedDenoiser
from nonmarkovian.vocab import IDX_TO_TOKEN


def build_synthetic_views(
    hat_x0: torch.Tensor,
    alphas: torch.Tensor,
    t_start: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Full trajectory q(x_tau | hat_x0) for tau = 0..T-1; Boltzmann router mixes views k > t_start."""
    T = alphas.shape[0]
    if not (0 <= t_start < T):
        raise ValueError("t_start out of range")
    views = []
    for tau in range(T):
        a = float(alphas[tau].item())
        views.append(corrupt_sequence(hat_x0, a, generator=generator))
    return torch.stack(views, dim=1)


@torch.no_grad()
def sample_sequences(
    model: RoutedDenoiser,
    alphas: torch.Tensor,
    batch: int,
    seq_len: int,
    device: torch.device,
    *,
    labels: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return sampled nucleotide indices [B, L] (0..3)."""
    model.eval()
    T = alphas.shape[0]
    hat = torch.zeros((batch, seq_len), device=device, dtype=torch.long)
    for t_start in range(T - 1, -1, -1):
        views = build_synthetic_views(hat, alphas, t_start, generator)
        logits, _pi, _h, _lb = model(views, t_start, labels=labels)
        hat = logits.argmax(dim=-1).clamp(max=3)
    return hat


def ids_to_strings(x: torch.Tensor, mask_pad: torch.Tensor | None = None) -> list[str]:
    out = []
    for i in range(x.shape[0]):
        chars = []
        for j in range(x.shape[1]):
            if mask_pad is not None and mask_pad[i, j]:
                break
            chars.append(IDX_TO_TOKEN[int(x[i, j].item())])
        out.append("".join(chars))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=200)
    p.add_argument("--label", type=int, default=-1, help="conditioning class if model has labels; -1 = none")
    p.add_argument(
        "--router_topk",
        type=int,
        default=0,
        help="Deprecated; Boltzmann router ignores this (kept for old configs)",
    )
    p.add_argument("--device", type=str, default="auto", help='"auto", "cpu", or "cuda"')
    p.add_argument("--out", type=str, default="samples.txt")
    args = p.parse_args()

    device = resolve_device_arg(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})
    num_timesteps = int(cfg.get("num_timesteps", 32))
    max_len = int(cfg.get("max_len", 256))
    seq_len = min(args.seq_len, max_len)
    num_classes = int(cfg.get("num_classes", 0))

    alphas = ckpt.get("alphas")
    if alphas is None:
        alphas = cosine_alpha_schedule(num_timesteps, device=device)
    else:
        alphas = alphas.to(device)

    cond_dim_raw = cfg.get("cond_dim", 0)
    cond_dim = int(cond_dim_raw) if cond_dim_raw else None
    if cond_dim == 0:
        cond_dim = None
    router_tau = float(cfg.get("router_tau", 1.0))
    router_k = int(cfg.get("router_k", 1))
    time_freq_dim = int(cfg.get("time_freq_dim", 256))
    # Legacy checkpoints stored enc_layers + dec_layers; new runs use dec_layers only.
    dec_layers_total = int(cfg.get("dec_layers", 6)) + int(cfg.get("enc_layers", 0))
    model = RoutedDenoiser(
        d_model=int(cfg.get("d_model", 256)),
        nhead=int(cfg.get("nhead", 8)),
        dec_layers=dec_layers_total,
        dim_ff=int(cfg.get("dim_ff", 1024)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_len=max_len,
        num_timesteps=num_timesteps,
        num_labels=num_classes if num_classes > 0 else None,
        cond_dim=cond_dim,
        router_tau=router_tau,
        router_k=router_k,
        time_freq_dim=time_freq_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    labels = None
    if num_classes > 0 and args.label >= 0:
        labels = torch.full((args.batch,), args.label, device=device, dtype=torch.long)

    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    x = sample_sequences(
        model,
        alphas,
        args.batch,
        seq_len,
        device,
        labels=labels,
        generator=gen,
    )
    lines = ids_to_strings(x.cpu())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")
    print(f"wrote {len(lines)} sequences to {out_path}")


if __name__ == "__main__":
    main()
