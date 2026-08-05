"""Reverse sampling for the protein routed DiT-BFN model.

Same SLM ``new_diff`` reverse process as ``SLM/slm.py:_sample_newdiff`` and
``nonmarkovian.sample.sample_sequences`` (Bernoulli transition on a simplex ``x_t``
with ``(x_t > 0)`` support masking), but generalised to the protein vocab and the
``RoutedDenoiserDiTBFN`` backbone. With ``history_mode="trajectory"`` the model's
``views`` tensor accumulates the reverse-process states, giving the non-Markovian
history conditioning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.model_protein import RoutedDenoiserDiTBFN
from nonmarkovian.sample import _expected_nums, _sample_bernoulli
from nonmarkovian.vocab_protein import VOCAB_SIZE, decode


@torch.no_grad()
def sample_protein_sequences(
    model: RoutedDenoiserDiTBFN,
    num_steps: int,
    batch: int,
    seq_len: int,
    device: torch.device,
    *,
    vocab: int = VOCAB_SIZE,
    bernoulli_scheduler: str = "loglinear",
    generator: torch.Generator | None = None,
    history_mode: str = "trajectory",
    corruption_mode: str = "independent",
    release_threshold: int = 6,
) -> torch.Tensor:
    """Generate ``[batch, seq_len]`` protein token ids via routed ``new_diff`` reverse sampling.

    ``release_threshold`` (tenths of ``num_steps``, enhancer ``sample.py`` parity) controls
    when the ``(x_t > 0)`` support-mask constraint is released in ``corruption_mode="independent"``:
    the mask is kept for the first ``release_threshold/10`` of the reverse steps and dropped
    afterwards, letting previously-zeroed categories re-activate in the final steps. In
    ``corruption_mode="trajectory"`` the support mask is always applied (monotone constraint).
    """
    model.eval()
    T = int(num_steps)
    x_t = torch.full((batch, seq_len, vocab), 1.0 / float(vocab), device=device, dtype=torch.float32)
    # history buffer: uniform everywhere; current x_t written into slot t_start each step.
    views_buffer = x_t.new_full((batch, T, seq_len, vocab), 1.0 / float(vocab))

    for i in range(1, T + 1):
        t = torch.full((batch, 1), 1.0 - float(i - 1) / float(T), device=device, dtype=torch.float32)
        t_start = T - i
        if history_mode == "uniform":
            views_buffer.fill_(1.0 / float(vocab))
        views_buffer[:, t_start] = x_t

        logits, _pi, _h, _lb, seq_in = model(
            views_buffer, t_start, t_cond=float(t[0, 0].item()), scheduler=bernoulli_scheduler
        )

        support_mask = (x_t > 0)
        has_any = support_mask.any(dim=-1, keepdim=True)
        support_mask = torch.where(has_any, support_mask, torch.ones_like(support_mask))
        logits = logits.masked_fill(~support_mask, torch.finfo(logits.dtype).min)
        model_prob = F.softmax(logits, dim=-1)
        model_prob = model_prob / model_prob.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        t3 = t.unsqueeze(-1)
        nominator = _expected_nums(t3 - 1.0 / float(T), num_classes=vocab, scheduler=bernoulli_scheduler) - 1.0
        denominator = torch.clamp(
            _expected_nums(t3, num_classes=vocab, scheduler=bernoulli_scheduler) - 1.0, min=1e-8
        )
        weight = torch.clamp(nominator / denominator, min=0.0, max=1.0)
        predicted = torch.clamp(model_prob + weight * (1.0 - model_prob), min=0.0, max=1.0)

        sample_pred = _sample_bernoulli(predicted, generator=generator)
        #release_threshold = 6
        print(release_threshold,"release_threshold",corruption_mode,"corruption_mode")
        if corruption_mode == "independent":
            # Keep support mask early, release it in the final steps (enhancer parity).
            if i <= release_threshold * T // 10:
                print("release_threshold",i)
                sample_pred = sample_pred & support_mask
            else:
                sample_pred = sample_pred 
        else:
            sample_pred = sample_pred & support_mask
        sample_pred_sum = sample_pred.sum(dim=-1, keepdim=True)
        fallback = F.one_hot(predicted.argmax(dim=-1), num_classes=vocab).to(dtype=torch.bool)
        sample_pred = torch.where(sample_pred_sum > 0, sample_pred, fallback)
        x_t = sample_pred.to(dtype=torch.float32)
        x_t = x_t / x_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    t_last = torch.full((batch, 1), 1.0 / float(T), device=device, dtype=torch.float32)
    if history_mode == "uniform":
        views_buffer.fill_(1.0 / float(vocab))
    views_buffer[:, 0] = x_t
    logits_last, _pi, _h, _lb, _seq_in = model(
        views_buffer, 0, t_cond=float(t_last[0, 0].item()), scheduler=bernoulli_scheduler
    )
    return logits_last.argmax(dim=-1).clamp(max=vocab - 1)


def _build_model_from_ckpt(cfg: dict, device) -> RoutedDenoiserDiTBFN:
    return RoutedDenoiserDiTBFN(
        vocab_size=int(cfg.get("vocab_size", VOCAB_SIZE)) if isinstance(cfg.get("vocab_size", None), int) else VOCAB_SIZE,
        max_len=int(cfg.get("max_len", 1024)),
        num_timesteps=int(cfg.get("num_timesteps", 500)),
        hidden_size=int(cfg.get("hidden_size", 480)),
        cond_dim=int(cfg.get("cond_dim", 128)),
        n_blocks=int(cfg.get("n_blocks", 12)),
        n_heads=int(cfg.get("n_heads", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
        scale_by_sigma=bool(cfg.get("scale_by_sigma", True)),
        embedding_nml=bool(cfg.get("embedding_nml", False)),
        entropy_condition=bool(cfg.get("entropy_condition", False)),
        router_tau=float(cfg.get("router_tau", 1.0)),
        router_k=int(cfg.get("router_k", 1)),
        router_conv_kernel=int(cfg.get("router_conv_kernel", 1)),
        router_out_channels=int(cfg.get("router_out_channels", 128)),
    ).to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=400, help="Generation length (SLM uniref samples 100-500).")
    p.add_argument("--num_steps", type=int, default=0, help="Reverse steps (0 = checkpoint num_timesteps).")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="protein_samples.txt")
    p.add_argument("--bernoulli_scheduler", type=str, default="loglinear", choices=("loglinear", "linear"))
    p.add_argument("--history_mode", type=str, default="trajectory", choices=("trajectory", "uniform"))
    p.add_argument("--corruption_mode", type=str, default="independent", choices=("independent", "trajectory"))
    p.add_argument(
        "--release_threshold", type=int, default=6,
        help="Release the (x_t>0) support mask after this many tenths of num_steps "
             "(independent mode only; enhancer sample.py uses 6 = release in the last 40%%).",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = resolve_device_arg(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})
    num_steps = args.num_steps if args.num_steps > 0 else int(cfg.get("num_timesteps", 500))

    model = _build_model_from_ckpt(cfg, device)
    model.num_timesteps = num_steps
    model.load_state_dict(ckpt["model"])

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    ids = sample_protein_sequences(
        model,
        num_steps,
        args.batch,
        args.seq_len,
        device,
        bernoulli_scheduler=args.bernoulli_scheduler,
        generator=gen,
        history_mode=args.history_mode,
        corruption_mode=args.corruption_mode,
        release_threshold=args.release_threshold,
    )
    lines = [decode(row.cpu()) for row in ids]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")
    print(f"wrote {len(lines)} protein sequences to {out_path}")


if __name__ == "__main__":
    main()
