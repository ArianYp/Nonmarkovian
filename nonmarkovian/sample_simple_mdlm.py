"""Sample from the baseline DiscreteDenoiser under the MDLM (masked / absorbing) process.

MDLM ancestral reverse with **carry-over unmasking** (the only coherent reverse for an
absorbing-state forward process):

  * start fully masked (all ``[M]``) at ``t = 1``;
  * at each reverse step ``t -> s`` (``s < t``) the model predicts ``p_theta(x_0 | x_t)``;
    each *currently-masked* position is unmasked with probability
    ``(alpha_s - alpha_t) / (1 - alpha_t)`` to a sample from that posterior;
  * already-unmasked positions are frozen (carry-over);
  * any positions still masked at the end are filled by the model's argmax at ``t -> 0``.

The model is imported and used **unchanged** from ``simple_model`` — masked tokens are mapped to
the uniform-1/4 simplex by ``tokens_to_four_channel_simplex`` inside the model's forward.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.forward_mdlm import mdlm_alpha
from nonmarkovian.sample import ids_to_strings
from nonmarkovian.simple_model import DiscreteDenoiser, DiscreteDenoiserCNN
from nonmarkovian.vocab import MASK_IDX


def _sample_categorical(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Categorical sample over the last dim. ``probs`` ``[B, L, C]`` -> ids ``[B, L]``."""
    B, L, C = probs.shape
    flat = probs.reshape(B * L, C).clamp(min=0.0)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    idx = torch.multinomial(flat, num_samples=1, generator=generator).squeeze(-1)
    return idx.view(B, L)


@torch.no_grad()
def sample_sequences_simple_mdlm(
    model: DiscreteDenoiser | DiscreteDenoiserCNN,
    num_steps: int,
    batch: int,
    seq_len: int,
    device: torch.device,
    *,
    num_timesteps_train: int | None = None,
    labels: torch.Tensor | None = None,
    guidance_scale: float = 0.0,
    scheduler: str = "loglinear",
    num_classes: int = 4,
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MDLM ancestral sampling (carry-over unmasking) for the simple denoiser.

    ``return_trajectory=True`` also returns the state trajectory ``[B, T + 2, L]`` and the
    unconstrained x0-belief trajectory ``[B, T, L]`` (both uint8, CPU), laid out exactly as in
    ``sample_mdlm.sample_sequences_mdlm``. Under carry-over the *states* can never be revised, so
    the belief trajectory is what carries the "changed its mind" signal here.
    """
    model.eval()
    T = int(num_steps)
    x = torch.full((batch, seq_len), MASK_IDX, device=device, dtype=torch.long)  # all masked @ t=1
    frames: list[torch.Tensor] | None = [] if return_trajectory else None
    pred_frames: list[torch.Tensor] | None = [] if return_trajectory else None
    if frames is not None:
        frames.append(x.to("cpu", torch.uint8))

    use_cfg = float(guidance_scale) != 0.0 and labels is not None
    null_lab: torch.Tensor | None = None
    if use_cfg:
        num_cls_for_null = getattr(model, "num_labels", None)
        if num_cls_for_null is not None and int(num_cls_for_null) > 0:
            null_lab = torch.full_like(labels, int(num_cls_for_null))
        else:
            null_lab = None  # DiT backbone uses None as the null path
    print(use_cfg, "use_cfg")
    for i in range(T):
        t_val = 1.0 - float(i) / float(T)        # current time  (1.0 -> 1/T)
        s_val = 1.0 - float(i + 1) / floathow (T)    # next time      (1-1/T -> 0)
        t_b = torch.full((batch,), t_val, device=device, dtype=torch.float32)

        if use_cfg:
            logits_c, _ = model(x, t_b, labels=labels)
            logits_u, _ = model(x, t_b, labels=null_lab)
            logits = (1.0 + guidance_scale) * logits_c - guidance_scale * logits_u
        else:
            logits, _ = model(x, t_b, labels=labels)
        if pred_frames is not None:
            pred_frames.append(logits[..., :num_classes].argmax(dim=-1).to("cpu", torch.uint8))
        probs = F.softmax(logits, dim=-1)                              # [B, L, C]

        # carry-over unmasking probability for a still-masked position
        t_t = torch.tensor(t_val, device=device)
        s_t = torch.tensor(s_val, device=device)
        a_t = mdlm_alpha(t_t, num_classes=num_classes, scheduler=scheduler)
        a_s = mdlm_alpha(s_t, num_classes=num_classes, scheduler=scheduler)
        denom = (1.0 - a_t).clamp(min=1e-8)
        unmask_prob = ((a_s - a_t) / denom).clamp(min=0.0, max=1.0)    # scalar

        is_masked = x == MASK_IDX
        if generator is None:
            u = torch.rand((batch, seq_len), device=device, dtype=torch.float32)
        else:
            u = torch.rand((batch, seq_len), device=device, dtype=torch.float32, generator=generator)
        do_unmask = is_masked & (u < unmask_prob)
        sampled = _sample_categorical(probs, generator=generator).clamp(max=num_classes - 1)
        x = torch.where(do_unmask, sampled, x)
        if frames is not None:
            frames.append(x.to("cpu", torch.uint8))

    # final clean-up: fill any positions still masked using the model's argmax at t -> 0
    if (x == MASK_IDX).any():
        t_last = torch.full((batch,), 1.0 / float(T), device=device, dtype=torch.float32)
        if use_cfg:
            logits_c, _ = model(x, t_last, labels=labels)
            logits_u, _ = model(x, t_last, labels=null_lab)
            logits_last = (1.0 + guidance_scale) * logits_c - guidance_scale * logits_u
        else:
            logits_last, _ = model(x, t_last, labels=labels)
        argmax_x0 = logits_last.argmax(dim=-1).clamp(max=num_classes - 1)
        x = torch.where(x == MASK_IDX, argmax_x0, x)

    x = x.clamp(max=num_classes - 1)
    if frames is not None:
        frames.append(x.to("cpu", torch.uint8))
        return x, torch.stack(frames, dim=1), torch.stack(pred_frames, dim=1)
    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=500)
    p.add_argument("--label", type=int, default=-1, help="conditioning class if model has labels; -1 = none")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="samples_simple_mdlm.txt")
    p.add_argument(
        "--num_timesteps_sample",
        type=int,
        default=0,
        help="Override number of reverse steps (0 = use checkpoint / --num_timesteps)",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default="loglinear",
        choices=("loglinear", "linear"),
        help="MDLM masking schedule (must match training).",
    )
    args = p.parse_args()

    device = resolve_device_arg(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})
    if ckpt.get("trainer") != "simple_mdlm" and cfg.get("trainer") != "simple_mdlm":
        print("Warning: checkpoint may not be from train_simple_mdlm.py (trainer != simple_mdlm).")

    num_timesteps_train = int(cfg.get("num_timesteps", 32))
    num_timesteps_sample = int(cfg.get("num_timesteps_sample", num_timesteps_train))
    if args.num_timesteps_sample > 0:
        num_timesteps_sample = int(args.num_timesteps_sample)
    max_len = int(cfg.get("max_len", 500))
    seq_len = min(args.seq_len, max_len)
    num_classes = int(cfg.get("num_classes", 0))
    scheduler = str(cfg.get("bernoulli_scheduler", args.scheduler))

    backbone = str(cfg.get("backbone", "dit")).lower()
    cnn_stacks = int(cfg.get("cnn_stacks", 4))
    dec_layers_total = int(cfg.get("dec_layers", 6)) + int(cfg.get("enc_layers", 0))
    if backbone == "cnn":
        model = DiscreteDenoiserCNN(
            d_model=int(cfg.get("d_model", 256)),
            max_len=max_len,
            num_timesteps=num_timesteps_train,
            num_labels=num_classes if num_classes > 0 else None,
            num_cnn_stacks=cnn_stacks,
        ).to(device)
    else:
        cond_dim_raw = cfg.get("cond_dim", 0)
        cond_dim = int(cond_dim_raw) if cond_dim_raw else None
        if cond_dim == 0:
            cond_dim = None
        time_freq_dim = int(cfg.get("time_freq_dim", 256))
        model = DiscreteDenoiser(
            d_model=int(cfg.get("d_model", 256)),
            nhead=int(cfg.get("nhead", 8)),
            dec_layers=dec_layers_total,
            dim_ff=int(cfg.get("dim_ff", 1024)),
            dropout=float(cfg.get("dropout", 0.1)),
            max_len=max_len,
            num_timesteps=num_timesteps_train,
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
    x = sample_sequences_simple_mdlm(
        model,
        num_timesteps_sample,
        args.batch,
        seq_len,
        device,
        num_timesteps_train=num_timesteps_train,
        labels=labels,
        scheduler=scheduler,
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
