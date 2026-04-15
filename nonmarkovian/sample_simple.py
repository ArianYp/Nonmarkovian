"""Sample from baseline DiscreteDenoiser (single x_t per step, no router)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.forward import corrupt_sequence, cosine_alpha_schedule
from nonmarkovian.simple_model import DiscreteDenoiser
from nonmarkovian.sample import ids_to_strings


@torch.no_grad()
def sample_sequences_simple(
    model: DiscreteDenoiser,
    alphas: torch.Tensor,
    batch: int,
    seq_len: int,
    device: torch.device,
    *,
    labels: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Iterative x0 prediction: at each t, corrupt current hat with alpha[t], predict x0."""
    model.eval()
    T = alphas.shape[0]
    hat = torch.zeros((batch, seq_len), device=device, dtype=torch.long)
    for t in range(T - 1, -1, -1):
        x_t = corrupt_sequence(hat, float(alphas[t].item()), generator=generator)
        t_b = torch.full((batch,), t, device=device, dtype=torch.long)
        logits, _ = model(x_t, t_b, labels=labels)
        hat = logits.argmax(dim=-1).clamp(max=3)
    return hat


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=500)
    p.add_argument("--label", type=int, default=-1, help="conditioning class if model has labels; -1 = none")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="samples_simple.txt")
    args = p.parse_args()

    device = resolve_device_arg(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})
    if ckpt.get("trainer") != "simple_discrete" and cfg.get("trainer") != "simple_discrete":
        print("Warning: checkpoint may not be from train_simple.py (trainer != simple_discrete).")

    num_timesteps = int(cfg.get("num_timesteps", 32))
    max_len = int(cfg.get("max_len", 500))
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
    time_freq_dim = int(cfg.get("time_freq_dim", 256))
    dec_layers_total = int(cfg.get("dec_layers", 6)) + int(cfg.get("enc_layers", 0))
    model = DiscreteDenoiser(
        d_model=int(cfg.get("d_model", 256)),
        nhead=int(cfg.get("nhead", 8)),
        dec_layers=dec_layers_total,
        dim_ff=int(cfg.get("dim_ff", 1024)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_len=max_len,
        num_timesteps=num_timesteps,
        num_labels=num_classes if num_classes > 0 else None,
        cond_dim=cond_dim,
        time_freq_dim=time_freq_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    labels = None
    if num_classes > 0 and args.label >= 0:
        labels = torch.full((args.batch,), args.label, device=device, dtype=torch.long)

    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    x = sample_sequences_simple(
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
