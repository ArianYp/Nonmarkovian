"""Sample from baseline DiscreteDenoiser (single x_t per step, no router)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.forward import cosine_alpha_schedule
from nonmarkovian.simple_model import DiscreteDenoiser, DiscreteDenoiserCNN
from nonmarkovian.sample import ids_to_strings


def _sample_bernoulli(categorical_probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    if generator is None:
        random_uniform_sample = torch.rand_like(categorical_probs)
    else:
        random_uniform_sample = torch.rand(
            categorical_probs.shape,
            device=categorical_probs.device,
            dtype=categorical_probs.dtype,
            generator=generator,
        )
    return random_uniform_sample < categorical_probs


def _expected_nums(t: torch.Tensor, *, num_classes: int = 4, scheduler: str = "loglinear") -> torch.Tensor:
    if scheduler == "loglinear":
        return torch.clamp(torch.exp(torch.log(torch.tensor(float(num_classes), device=t.device)) * t), min=1.0)
    if scheduler == "linear":
        return torch.clamp(float(num_classes) * t, min=1.0)
    raise ValueError(f"Unknown Bernoulli scheduler: {scheduler}")


@torch.no_grad()
def sample_sequences_simple(
    model: DiscreteDenoiser | DiscreteDenoiserCNN,
    alphas_sample: torch.Tensor,
    batch: int,
    seq_len: int,
    device: torch.device,
    *,
    num_timesteps_train: int | None = None,
    labels: torch.Tensor | None = None,
    guidance_scale: float = 0.0,
    bernoulli_scheduler: str = "loglinear",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """SLM-style new_diff reverse sampling in simplex space."""
    model.eval()
    num_steps = int(alphas_sample.shape[0])
    vocab = 4
    x_t = torch.full((batch, seq_len, vocab), 1.0 / float(vocab), device=device, dtype=torch.float32)

    use_cfg = float(guidance_scale) != 0.0 and labels is not None
    null_lab: torch.Tensor | None = None
    if use_cfg:
        num_cls_for_null = getattr(model, "num_labels", None)
        if num_cls_for_null is not None and int(num_cls_for_null) > 0:
            null_lab = torch.full_like(labels, int(num_cls_for_null))
        else:
            null_lab = None  # DiT backbone uses None as the null path

    for i in range(1, num_steps + 1):
        t = torch.full((batch, 1), 1.0 - float(i - 1) / float(num_steps), device=device, dtype=torch.float32)
        if use_cfg:
            logits_c, _ = model(x_t, t.squeeze(-1), labels=labels)
            logits_u, _ = model(x_t, t.squeeze(-1), labels=null_lab)
            logits = (1.0 + guidance_scale) * logits_c - guidance_scale * logits_u
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        else:
            logits, _ = model(x_t, t.squeeze(-1), labels=labels)

        support_mask = x_t > 0
        has_any = support_mask.any(dim=-1, keepdim=True)
        support_mask = torch.where(has_any, support_mask, torch.ones_like(support_mask))
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~support_mask, neg_inf)

        model_prob = F.softmax(logits, dim=-1)
        if not torch.allclose(
            model_prob.sum(dim=-1),
            torch.ones((batch, seq_len), device=device, dtype=model_prob.dtype),
            atol=1e-4,
        ):
            model_prob = model_prob / model_prob.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        t3 = t.unsqueeze(-1)
        nominator = _expected_nums(t3 - 1.0 / float(num_steps), scheduler=bernoulli_scheduler) - 1.0
        denominator = torch.clamp(_expected_nums(t3, scheduler=bernoulli_scheduler) - 1.0, min=1e-8)
        weight = torch.clamp(nominator / denominator, min=0.0, max=1.0)
        predicted = torch.clamp(model_prob + weight * (1.0 - model_prob), min=0.0, max=1.0)

        sample_pred = _sample_bernoulli(predicted, generator=generator) & (x_t > 0)
        sample_pred_sum = sample_pred.sum(dim=-1, keepdim=True)
        fallback = F.one_hot(predicted.argmax(dim=-1), num_classes=vocab).to(dtype=torch.bool)
        sample_pred = torch.where(sample_pred_sum > 0, sample_pred, fallback)
        x_t = sample_pred.to(dtype=torch.float32)
        x_t = x_t / x_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    t_last = torch.full((batch, 1), 1.0 / float(num_steps), device=device, dtype=torch.float32)
    if use_cfg:
        logits_c, _ = model(x_t, t_last.squeeze(-1), labels=labels)
        logits_u, _ = model(x_t, t_last.squeeze(-1), labels=null_lab)
        logits_last = (1.0 + guidance_scale) * logits_c - guidance_scale * logits_u
        logits_last = logits_last - torch.logsumexp(logits_last, dim=-1, keepdim=True)
    else:
        logits_last, _ = model(x_t, t_last.squeeze(-1), labels=labels)
    return logits_last.argmax(dim=-1).clamp(max=3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=500)
    p.add_argument("--label", type=int, default=-1, help="conditioning class if model has labels; -1 = none")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="samples_simple.txt")
    p.add_argument(
        "--num_timesteps_sample",
        type=int,
        default=0,
        help="Override number of reverse steps (0 = use checkpoint / --num_timesteps)",
    )
    p.add_argument(
        "--bernoulli_scheduler",
        type=str,
        default="loglinear",
        choices=("loglinear", "linear"),
        help="SLM-style Bernoulli scheduler for new_diff sampling.",
    )
    args = p.parse_args()

    device = resolve_device_arg(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})
    if ckpt.get("trainer") != "simple_discrete" and cfg.get("trainer") != "simple_discrete":
        print("Warning: checkpoint may not be from train_simple.py (trainer != simple_discrete).")

    num_timesteps_train = int(cfg.get("num_timesteps", 32))
    num_timesteps_sample = int(cfg.get("num_timesteps_sample", num_timesteps_train))
    if args.num_timesteps_sample > 0:
        num_timesteps_sample = int(args.num_timesteps_sample)
    max_len = int(cfg.get("max_len", 500))
    seq_len = min(args.seq_len, max_len)
    num_classes = int(cfg.get("num_classes", 0))

    alphas_sample = ckpt.get("alphas_sample")
    if alphas_sample is None:
        alphas_sample = cosine_alpha_schedule(num_timesteps_sample, device=device)
    else:
        alphas_sample = alphas_sample.to(device)
        if alphas_sample.shape[0] != num_timesteps_sample:
            alphas_sample = cosine_alpha_schedule(num_timesteps_sample, device=device)

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
    x = sample_sequences_simple(
        model,
        alphas_sample,
        args.batch,
        seq_len,
        device,
        num_timesteps_train=num_timesteps_train,
        labels=labels,
        bernoulli_scheduler=args.bernoulli_scheduler,
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
