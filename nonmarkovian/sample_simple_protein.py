"""Reverse sampling for the BASELINE (non-routed) protein DiT-BFN model.

Exact SLM ``new_diff`` reverse process (``SLM/slm.py:_sample_newdiff``): a single simplex
``x_t`` denoised step by step with ``(x_t > 0)`` support masking -- no history / router.
Counterpart of ``nonmarkovian.sample_protein`` for method comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.sample import _expected_nums, _sample_bernoulli
from nonmarkovian.simple_model_protein import DiscreteDenoiserDiTBFN
from nonmarkovian.vocab_protein import VOCAB_SIZE, decode


@torch.no_grad()
def sample_simple_protein(
    model: DiscreteDenoiserDiTBFN,
    num_steps: int,
    batch: int,
    seq_len: int,
    device: torch.device,
    *,
    vocab: int = VOCAB_SIZE,
    bernoulli_scheduler: str = "loglinear",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate ``[batch, seq_len]`` protein token ids via baseline ``new_diff`` reverse sampling."""
    model.eval()
    T = int(num_steps)
    x_t = torch.full((batch, seq_len, vocab), 1.0 / float(vocab), device=device, dtype=torch.float32)
    for i in range(1, T + 1):
        t = torch.full((batch, 1), 1.0 - float(i - 1) / float(T), device=device, dtype=torch.float32)
        logits, _h = model(x_t, t.squeeze(-1))
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

        sample_pred = _sample_bernoulli(predicted, generator=generator) & support_mask
        sample_pred_sum = sample_pred.sum(dim=-1, keepdim=True)
        fallback = F.one_hot(predicted.argmax(dim=-1), num_classes=vocab).to(dtype=torch.bool)
        sample_pred = torch.where(sample_pred_sum > 0, sample_pred, fallback)
        x_t = sample_pred.to(dtype=torch.float32)
        x_t = x_t / x_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    t_last = torch.full((batch,), 1.0 / float(T), device=device, dtype=torch.float32)
    logits_last, _h = model(x_t, t_last)
    return logits_last.argmax(dim=-1).clamp(max=vocab - 1)


def _build_model_from_ckpt(cfg: dict, device) -> DiscreteDenoiserDiTBFN:
    return DiscreteDenoiserDiTBFN(
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
    ).to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=400)
    p.add_argument("--num_steps", type=int, default=0, help="Reverse steps (0 = checkpoint num_timesteps).")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="protein_simple_samples.txt")
    p.add_argument("--bernoulli_scheduler", type=str, default="loglinear", choices=("loglinear", "linear"))
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
    ids = sample_simple_protein(
        model, num_steps, args.batch, args.seq_len, device,
        bernoulli_scheduler=args.bernoulli_scheduler, generator=gen,
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
