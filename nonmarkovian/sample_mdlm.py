"""Iterative MDLM (masked / absorbing) sampling for the routed non-Markovian model.

Structurally mirrors ``sample.sample_sequences``: the denoiser receives a ``views`` tensor
``[B, T, L, 4]`` built from the running reverse trajectory (see ``history_mode``), and the router
mixes the current view with selected history. The **only** difference from ``sample.py`` is the
per-step update: instead of the Bernoulli new_diff reverse step, this uses MDLM ancestral
**carry-over unmasking** (start all ``[M]``; progressively unmask masked positions to samples from
``p_theta(x_0 | x_t)`` with prob ``(alpha_s - alpha_t)/(1 - alpha_t)``; frozen once unmasked).

The model (``RoutedDenoiserCNN``) is imported and used **unchanged** — masked tokens become the
uniform-1/4 simplex via ``tokens_to_four_channel_simplex``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.forward_mdlm import mdlm_alpha, mdlm_mask_prob
from nonmarkovian.model import RoutedDenoiserCNN
from nonmarkovian.slm_cnn import tokens_to_four_channel_simplex
from nonmarkovian.vocab import IDX_TO_TOKEN, MASK_IDX


def _sample_categorical(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    B, L, C = probs.shape
    flat = probs.reshape(B * L, C).clamp(min=0.0)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    idx = torch.multinomial(flat, num_samples=1, generator=generator).squeeze(-1)
    return idx.view(B, L)


@torch.no_grad()
def sample_sequences_mdlm(
    model: RoutedDenoiserCNN,
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
    history_mode: str = "trajectory",
    corruption_mode: str = "independent",
    independent_threshold: float = 0.6,
) -> torch.Tensor:
    """MDLM ancestral sampling with the routed history-aware denoiser.

    The ``views_buffer`` accumulates the running reverse-process simplices exactly as in
    ``sample.sample_sequences`` (``history_mode='trajectory'``): at step ``i`` the current
    ``x_t`` (mask -> uniform 1/4) is written into slot ``t_start = num_steps - i``, so slots at
    indices ``> t_start`` hold past (more-masked) states -- genuine history. ``'uniform'`` keeps
    non-current slots at ``1/C``.

    Reverse-step constraint (mirrors ``sample.py``'s Bernoulli ``corruption_mode`` / threshold):

    - ``corruption_mode='trajectory'`` (Markovian): strict **carry-over** -- once a position is
      unmasked it is frozen; only still-masked positions may be filled. Same as the simple model.
    - ``corruption_mode='independent'`` (non-Markovian, default): carry-over holds only for the
      first ``independent_threshold`` fraction of steps; **after** that the constraint is dropped
      and each step samples x0_hat then re-corrupts to the next time (``q(x_s | x0_hat)``), so a
      previously-**unmasked token can be masked again** (and vice-versa). This freedom to re-mask /
      undo earlier commitments is the whole point of the non-Markovian variant. (Contrast the
      simple/Markovian model, where once unmasked a token stays unmasked forever.)
    """
    model.eval()
    T = int(num_steps)
    C = int(num_classes)

    # Align the model's internal timestep count to num_steps (mirrors sample.py).
    old_num_timesteps = getattr(model, "num_timesteps", None)
    model.num_timesteps = T

    use_cfg = float(guidance_scale) != 0.0 and labels is not None
    null_lab: torch.Tensor | None = None
    if use_cfg:
        num_cls_for_null = getattr(model, "num_labels", None)
        if num_cls_for_null is not None and int(num_cls_for_null) > 0:
            null_lab = torch.full_like(labels, int(num_cls_for_null))
        else:
            null_lab = None

    x = torch.full((batch, seq_len), MASK_IDX, device=device, dtype=torch.long)  # all masked @ t=1
    views_buffer = torch.full((batch, T, seq_len, C), 1.0 / C, device=device, dtype=torch.float32)
    print(independent_threshold, "use_cfg", use_cfg)
    for i in range(1, T + 1):
        t_val = 1.0 - float(i - 1) / float(T)   # current time   (1.0 -> 1/T)
        s_val = 1.0 - float(i) / float(T)        # next time       (1-1/T -> 0)
        t_start = T - i                          # T-1 -> 0

        if history_mode == "uniform":
            views_buffer.fill_(1.0 / C)
        views_buffer[:, t_start] = tokens_to_four_channel_simplex(x)

        if use_cfg:
            logits_c, _pi, _h, _lb, _seq = model(
                views_buffer, t_start, labels=labels, t_cond=t_val
            )
            logits_u, _, _, _, _ = model(
                views_buffer, t_start, labels=null_lab, t_cond=t_val
            )
            logits = (1.0 + guidance_scale) * logits_c - guidance_scale * logits_u
        else:
            logits, _pi, _h, _lb, _seq = model(
                views_buffer, t_start, labels=labels, t_cond=t_val
            )
        support_mask = (_seq > 0) if _seq is not None else (x > 0)
        support_mask = (_seq >0) 
        support_mask = (tokens_to_four_channel_simplex(x) > 0)

        has_any = support_mask.any(dim=-1, keepdim=True)
        support_mask = torch.where(has_any, support_mask, torch.ones_like(support_mask))
        neg_inf = torch.finfo(logits.dtype).min   
        logits = logits.masked_fill(~support_mask, neg_inf)
        probs = F.softmax(logits, dim=-1)
        #print(probs.shape, "probs.shape")
        t_t = torch.tensor(t_val, device=device)
        s_t = torch.tensor(s_val, device=device)
        a_t = mdlm_alpha(t_t, num_classes=C, scheduler=scheduler)
        a_s = mdlm_alpha(s_t, num_classes=C, scheduler=scheduler)
        unmask_prob = ((a_s - a_t) / (1.0 - a_t).clamp(min=1e-8)).clamp(min=0.0, max=1.0)

        if generator is None:
            u = torch.rand((batch, seq_len), device=device, dtype=torch.float32)
        else:
            u = torch.rand((batch, seq_len), device=device, dtype=torch.float32, generator=generator)
        sampled = _sample_categorical(probs, generator=generator).clamp(max=C - 1)  # predicted x0_hat
        #print(f"i: {i}, independent_threshold: {independent_threshold}, T: {T}")
        #independent_threshold = 0.8
        if corruption_mode == "independent" and i > independent_threshold * T:
            # Non-Markovian corrector phase: drop the carry-over constraint. Sample the predicted
            # clean sequence x0_hat, then RE-CORRUPT to the next time s ~ q(x_s | x0_hat): each
            # position is independently re-masked with prob mask_prob(s). So a previously-UNMASKED
            # token can be MASKED AGAIN (and vice-versa) -- the model may undo/revise earlier
            # commitments. mask_prob(s) -> 0 as s -> 0, so this anneals back to a clean sequence.
            p_s = mdlm_mask_prob(s_t, num_classes=C, scheduler=scheduler)
            remask = u < p_s
            x = torch.where(remask, torch.full_like(x, MASK_IDX), sampled)
        else:
            # Carry-over (Markovian): only fill still-masked positions; keep resolved ones frozen.
            is_masked = x == MASK_IDX
            do_unmask = is_masked & (u < unmask_prob)
            x = torch.where(do_unmask, sampled, x)

    # final clean-up: any still-masked positions -> argmax at t -> 0.
    if (x == MASK_IDX).any():
        t_last = 1.0 / float(T)
        if history_mode == "uniform":
            views_buffer.fill_(1.0 / C)
        views_buffer[:, 0] = tokens_to_four_channel_simplex(x)
        if use_cfg:
            logits_c, _pi, _h, _lb, _seq = model(views_buffer, 0, labels=labels, t_cond=t_last)
            logits_u, _, _, _, _ = model(views_buffer, 0, labels=null_lab, t_cond=t_last)
            logits_last = (1.0 + guidance_scale) * logits_c - guidance_scale * logits_u
        else:
            logits_last, _pi, _h, _lb, _seq = model(views_buffer, 0, labels=labels, t_cond=t_last)
        argmax_x0 = logits_last.argmax(dim=-1).clamp(max=C - 1)
        x = torch.where(x == MASK_IDX, argmax_x0, x)

    if old_num_timesteps is not None:
        model.num_timesteps = old_num_timesteps
    return x.clamp(max=C - 1)


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
    p.add_argument("--seq_len", type=int, default=500)
    p.add_argument("--label", type=int, default=-1, help="conditioning class if model has labels; -1 = none")
    p.add_argument("--device", type=str, default="auto", help='"auto", "cpu", or "cuda"')
    p.add_argument("--out", type=str, default="samples_mdlm.txt")
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
    p.add_argument(
        "--history_mode",
        type=str,
        default="trajectory",
        choices=("trajectory", "uniform"),
        help="How to fill non-current view slots during reverse sampling.",
    )
    p.add_argument("--guidance_scale", type=float, default=0.0)
    p.add_argument(
        "--corruption_mode",
        type=str,
        default="",
        choices=("", "independent", "trajectory"),
        help="Reverse-step constraint: 'trajectory' = strict carry-over (Markovian); "
        "'independent' = drop carry-over after --independent_threshold (non-Markovian). "
        "Empty = use the checkpoint's training value.",
    )
    p.add_argument(
        "--independent_threshold",
        type=float,
        default=-1.0,
        help="Fraction of reverse steps after which the carry-over constraint is dropped "
        "(independent mode). <0 = use the checkpoint's value (default 0.6).",
    )
    args = p.parse_args()

    device = resolve_device_arg(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})
    if ckpt.get("trainer") != "routed_mdlm" and cfg.get("trainer") != "routed_mdlm":
        print("Warning: checkpoint may not be from train_mdlm.py (trainer != routed_mdlm).")

    num_timesteps_train = int(cfg.get("num_timesteps", 32))
    num_timesteps_sample = int(cfg.get("num_timesteps_sample", num_timesteps_train))
    if args.num_timesteps_sample > 0:
        num_timesteps_sample = int(args.num_timesteps_sample)
    max_len = int(cfg.get("max_len", 500))
    seq_len = min(args.seq_len, max_len)
    num_classes = int(cfg.get("num_classes", 0))
    scheduler = str(cfg.get("bernoulli_scheduler", args.scheduler))
    corruption_mode = args.corruption_mode or str(cfg.get("corruption_mode", "independent"))
    independent_threshold = (
        args.independent_threshold if args.independent_threshold >= 0.0
        else float(cfg.get("independent_threshold", 0.6))
    )

    backbone = str(cfg.get("backbone", "cnn")).lower()
    if backbone != "cnn":
        raise NotImplementedError("sample_mdlm only supports the CNN backbone (enhancer setup).")
    cnn_stacks = int(cfg.get("cnn_stacks", 4))
    model = RoutedDenoiserCNN(
        d_model=int(cfg.get("d_model", 256)),
        max_len=max_len,
        num_timesteps=num_timesteps_sample,
        num_labels=num_classes if num_classes > 0 else None,
        router_tau=float(cfg.get("router_tau", 1.0)),
        router_k=int(cfg.get("router_k", 1)),
        num_cnn_stacks=cnn_stacks,
        router_conv_kernel=int(cfg.get("router_conv_kernel", 3)),
        router_out_channels=int(cfg.get("router_out_channels", 128)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.num_timesteps = num_timesteps_sample

    labels = None
    if num_classes > 0 and args.label >= 0:
        labels = torch.full((args.batch,), args.label, device=device, dtype=torch.long)

    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    x = sample_sequences_mdlm(
        model,
        num_timesteps_sample,
        args.batch,
        seq_len,
        device,
        num_timesteps_train=num_timesteps_train,
        labels=labels,
        guidance_scale=float(args.guidance_scale),
        scheduler=scheduler,
        generator=gen,
        history_mode=args.history_mode,
        corruption_mode=corruption_mode,
        independent_threshold=independent_threshold,
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
