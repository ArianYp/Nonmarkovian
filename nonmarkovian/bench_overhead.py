"""Compute-overhead ablation for the **SLM** (Bernoulli-simplex) non-Markovian path.

Answers "what does the router / history machinery cost?" with no dataset and no
checkpoint: synthetic ``x0``, randomly initialised models, and the *exact* train
step and samplers used by ``train.py`` / ``train_simple.py`` / ``sample.py`` /
``sample_simple.py``.

All variants are measured **round-robin** (one step of every variant per
iteration) so no single variant absorbs the allocator / cuDNN-autotune warmup --
variant-by-variant timing loops give the first variant a large systematic
penalty and are not comparable.

Variants measured
-----------------
Training (one optimizer step, CUDA-synced, split into views / forward / loss / backward):

* ``baseline``          -- ``DiscreteDenoiserCNN`` + single-view
  ``corrupt_sequence_bernoulli`` (the Markovian control, i.e. ``train_simple.py``).
* ``routed@t_start=..`` -- ``RoutedDenoiserCNN`` + ``sample_all_views_bernoulli``
  views ``[B, T-t_start, L, 4]``, i.e. ``train.py`` at a given ``t_start``
  (candidates ``K = T - t_start - 1``). ``t_start=0`` is the worst case; a real
  run draws ``t_start`` uniformly, so its expected cost sits near ``K=(T-1)/2``.
* ``routed_K0``         -- the *same routed module* forced to ``K=0`` so
  ``forward`` takes the ``z_cand is None`` branch. Router and candidate mixing
  are skipped while params, wrapper and denoiser stay identical, so
  ``routed@t_start=0`` minus ``routed_K0`` is the router's cost alone.

With ``--split_forward`` a pre-hook on ``model.cnn`` splits the routed forward into
*router* (everything before the denoiser) and *denoiser* time.

Sampling (reverse process, matched NFE per step):

* ``baseline``            -- ``sample_sequences_simple``.
* ``routed/trajectory``   -- ``sample_sequences`` with real history.
* ``routed/uniform``      -- ``sample_sequences`` with ``history_mode="uniform"``:
  identical FLOPs and memory, zero history information. This is the
  compute-matched control for "is the gain just extra compute?".

Also reported: parameter counts (router subset vs total), analytic FLOPs for the
router vs the CNN denoiser, the views-buffer footprint, and peak CUDA memory.

Example (production setting: T=1000 train, 10 sampling steps, rk=1, C_out=256)::

    python -m nonmarkovian.bench_overhead \
      --device cuda --batch_size 64 --seq_len 500 --num_timesteps 1000 \
      --cnn_stacks 4 --router_conv_kernel 1 --router_out_channels 256 \
      --t_start_frac 0.0 0.5 0.9 --split_forward \
      --sample_steps 10 --sample_batch 64 \
      --iters 20 --warmup 5 --csv logs/bench_overhead_slm.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from nonmarkovian.forward import (
    corrupt_sequence_bernoulli,
    cosine_alpha_schedule,
    sample_all_views_bernoulli,
)
from nonmarkovian.model import RoutedDenoiserCNN
from nonmarkovian.sample import sample_sequences
from nonmarkovian.sample_simple import sample_sequences_simple
from nonmarkovian.simple_model import DiscreteDenoiserCNN
from nonmarkovian.train_timing import sync_device, tic, toc_ms

ROUTER_PARAM_PREFIXES = ("W_cur", "W_phi", "state_router_proj")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _params(model: torch.nn.Module) -> tuple[int, int]:
    """(total params, router-only params)."""
    total = sum(p.numel() for p in model.parameters())
    router = sum(
        p.numel() for n, p in model.named_parameters() if n.startswith(ROUTER_PARAM_PREFIXES)
    )
    return total, router


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak_mem_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024.0**2)


class _CnnEntryProbe:
    """Forward pre-hook recording a device-synced timestamp when ``cnn`` is entered.

    Lets us attribute the routed forward to *router* (before the hook) vs
    *denoiser* (after). The sync makes attribution exact at the price of one
    extra stall per forward, so this is opt-in (``--split_forward``).
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.t_entry: float | None = None

    def __call__(self, module, inputs):  # noqa: ANN001 - torch hook signature
        sync_device(self.device)
        self.t_entry = time.perf_counter()


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def analytic_flops(
    *,
    batch: int,
    seq_len: int,
    num_candidates: int,
    router_conv_kernel: int,
    router_out_channels: int,
    cnn_stacks: int,
    hidden_dim: int = 128,
    vocab: int = 4,
) -> dict[str, float]:
    """MAC counts (multiply-accumulates, forward only) for router vs denoiser.

    Router, ``rk == 1`` (the collapsed ``M = W_cur^T W_phi`` fast path in
    ``_compatibility_scores_full_sequence``): ``M`` costs ``4*C_out*4``,
    ``zt_proj`` costs ``B*L*4*4`` and the score einsum ``B*K*L*4`` -- note that
    ``C_out`` drops out, so a wide router is free at ``rk=1``.

    Router, ``rk > 1``: two conv1d maps, ``B*C_out*L*4*rk`` for the current view
    and ``B*K*C_out*L*4*rk`` for the candidates, plus ``B*K*C_out*L`` for the dot
    product -- linear in ``K``, ``C_out`` *and* ``rk``.

    Denoiser: ``5*cnn_stacks`` conv1d layers of ``H*H*9`` per position, plus the
    input conv and the ``final_conv`` 1x1 pair.
    """
    B, L, K = float(batch), float(seq_len), float(num_candidates)
    rk, c_out = float(router_conv_kernel), float(router_out_channels)
    if router_conv_kernel == 1:
        router = vocab * c_out * vocab + B * L * vocab * vocab + B * K * L * vocab
    else:
        router = B * c_out * L * vocab * rk + B * K * c_out * L * vocab * rk + B * K * c_out * L
    mixing = B * K * L * vocab  # pi-weighted sum over candidates
    n_layers = 5.0 * float(cnn_stacks)
    H = float(hidden_dim)
    denoiser = (
        B * L * vocab * H * 9.0  # input conv (kernel 9)
        + n_layers * B * L * H * H * 9.0  # residual stack
        + n_layers * B * H * H  # per-layer time/cls Dense
        + B * L * H * H  # final_conv 1x1
        + B * L * H * vocab  # final_conv 1x1 -> logits
    )
    return {
        "router_macs": router + mixing,
        "denoiser_macs": denoiser,
        "router_frac_pct": 100.0 * (router + mixing) / denoiser,
    }


def _loss_from_logits(logits: torch.Tensor, x0: torch.Tensor, T: int, without_T: bool) -> torch.Tensor:
    """Exactly the diffusion NLL used by train.py / train_simple.py."""
    target = x0.clamp(max=3)
    log_probs = F.log_softmax(logits, dim=-1)
    nlog_p = -torch.gather(log_probs, -1, target[:, :, None]).squeeze(-1)
    if not without_T:
        nlog_p = float(T) * nlog_p
    denom = float(target.shape[0] * target.shape[1])
    return nlog_p.float().sum() / denom


# --------------------------------------------------------------------------- #
# one measured training step per variant
# --------------------------------------------------------------------------- #
class TrainVariant:
    """One measured optimizer step, mirroring the corresponding trainer."""

    def __init__(
        self,
        name: str,
        *,
        model: torch.nn.Module,
        opt: torch.optim.Optimizer,
        routed: bool,
        t_start: int,
        args,
        device: torch.device,
        probe: _CnnEntryProbe | None,
    ) -> None:
        self.name = name
        self.model = model
        self.opt = opt
        self.routed = routed
        self.t_start = t_start
        self.args = args
        self.device = device
        self.probe = probe
        self.K = max(args.num_timesteps - t_start - 1, 0) if routed else 0
        self.samples: list[dict[str, float]] = []
        self.peak_mem_mb = 0.0

    def step(self, x0: torch.Tensor) -> dict[str, float]:
        args, device, T = self.args, self.device, self.args.num_timesteps
        rec: dict[str, float] = {}

        t0 = tic(device)
        if self.routed:
            model_in = sample_all_views_bernoulli(
                x0,
                T,
                t_start=self.t_start,
                scheduler=args.bernoulli_scheduler,
                corruption_mode=args.corruption_mode,
            )
        else:
            t_cont_b = torch.full(
                (x0.shape[0], 1), float(self.t_start + 1) / float(T), device=device, dtype=torch.float32
            )
            model_in = corrupt_sequence_bernoulli(x0, t_cont_b, scheduler=args.bernoulli_scheduler)
        rec["ms_views"] = toc_ms(t0, device)

        t0 = tic(device)
        if self.probe is not None:
            self.probe.t_entry = None
        loss_bal: torch.Tensor | None = None
        if self.routed:
            logits, _pi, _h, loss_bal, _seq_in = self.model(
                model_in, 0, labels=None, t_cond=float(self.t_start + 1) / float(T),
                t_start_abs=self.t_start,
            )
        else:
            logits, _h = self.model(model_in, t_cont_b.squeeze(-1), labels=None)
        rec["ms_forward"] = toc_ms(t0, device)
        if self.probe is not None and self.probe.t_entry is not None:
            rec["ms_forward_router"] = (self.probe.t_entry - t0) * 1000.0
            rec["ms_forward_denoiser"] = rec["ms_forward"] - rec["ms_forward_router"]

        t0 = tic(device)
        loss = _loss_from_logits(logits, x0, T, args.without_T)
        if loss_bal is not None and args.router_lambda_bal > 0:
            loss = loss + args.router_lambda_bal * loss_bal
        rec["ms_loss"] = toc_ms(t0, device)

        t0 = tic(device)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        rec["ms_backward"] = toc_ms(t0, device)

        rec["ms_step"] = rec["ms_views"] + rec["ms_forward"] + rec["ms_loss"] + rec["ms_backward"]
        return rec

    def summary(self, x0: torch.Tensor) -> dict:
        keys = [
            "ms_views", "ms_forward", "ms_forward_router", "ms_forward_denoiser",
            "ms_loss", "ms_backward", "ms_step",
        ]
        out: dict = {"phase": "train", "variant": self.name, "K": self.K}
        for k in keys:
            vals = [s[k] for s in self.samples if k in s]
            if vals:
                out[k] = _mean(vals)
        out["ms_step_std"] = _std([s["ms_step"] for s in self.samples])
        out["steps_per_s"] = 1000.0 / out["ms_step"] if out.get("ms_step") else float("nan")
        out["peak_mem_mb"] = self.peak_mem_mb if self.peak_mem_mb > 0 else float("nan")
        if self.routed:
            out["views_buffer_mb"] = (
                x0.shape[0] * (self.K + 1) * x0.shape[1] * 4 * 4 / (1024.0**2)
            )
            out.update(
                analytic_flops(
                    batch=x0.shape[0],
                    seq_len=x0.shape[1],
                    num_candidates=self.K,
                    router_conv_kernel=self.args.router_conv_kernel,
                    router_out_channels=self.args.router_out_channels,
                    cnn_stacks=self.args.cnn_stacks,
                )
            )
        return out


class SampleVariant:
    """One measured reverse-sampling run."""

    def __init__(
        self,
        name: str,
        *,
        model: torch.nn.Module,
        routed: bool,
        history_mode: str,
        args,
        device: torch.device,
    ) -> None:
        self.name = name
        self.model = model
        self.routed = routed
        self.history_mode = history_mode
        self.args = args
        self.device = device
        self.alphas = cosine_alpha_schedule(args.sample_steps, device=device)
        self.gen = torch.Generator(device=device)
        self.times: list[float] = []
        self.peak_mem_mb = 0.0

    def step(self, it: int) -> float:
        args, device = self.args, self.device
        self.gen.manual_seed(1234 + it)
        t0 = tic(device)
        if self.routed:
            sample_sequences(
                self.model,
                self.alphas,
                args.sample_batch,
                args.seq_len,
                device,
                num_timesteps_train=args.num_timesteps,
                labels=None,
                bernoulli_scheduler=args.bernoulli_scheduler,
                generator=self.gen,
                history_mode=self.history_mode,
                corruption_mode=args.corruption_mode,
            )
        else:
            sample_sequences_simple(
                self.model,
                self.alphas,
                args.sample_batch,
                args.seq_len,
                device,
                num_timesteps_train=args.num_timesteps,
                labels=None,
                bernoulli_scheduler=args.bernoulli_scheduler,
                generator=self.gen,
            )
        return toc_ms(t0, device)

    def summary(self) -> dict:
        m = _mean(self.times)
        return {
            "phase": "sample",
            "variant": self.name,
            "K": self.args.sample_steps - 1,
            "ms_batch": m,
            "ms_batch_std": _std(self.times),
            "ms_per_step": m / float(self.args.sample_steps),
            "ms_per_seq": m / float(self.args.sample_batch),
            "seq_per_s": 1000.0 * self.args.sample_batch / m if m > 0 else float("nan"),
            "peak_mem_mb": self.peak_mem_mb if self.peak_mem_mb > 0 else float("nan"),
        }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:
            return "n/a"
        return f"{v:.4g}" if (abs(v) >= 1e6 or (v != 0 and abs(v) < 1e-2)) else f"{v:.2f}"
    return str(v)


def _print_table(rows: list[dict], cols: list[str], title: str) -> None:
    present = [c for c in cols if any(c in r for r in rows)]
    widths = {c: max([len(c)] + [len(_fmt(r.get(c, ""))) for r in rows]) for c in present}
    header = "  ".join(c.rjust(widths[c]) for c in present)
    print(f"\n{title}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r.get(c, "")).rjust(widths[c]) for c in present))
    print("-" * len(header))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=500)
    p.add_argument("--num_timesteps", type=int, default=1000, help="Training T (drives K and the views buffer)")
    p.add_argument("--cnn_stacks", type=int, default=4)
    p.add_argument("--d_model", type=int, default=32, help="Unused by the CNN backbone; kept for ctor parity")
    p.add_argument("--router_tau", type=float, default=0.01)
    p.add_argument("--router_k", type=int, default=2)
    p.add_argument("--router_conv_kernel", type=int, default=1)
    p.add_argument("--router_out_channels", type=int, default=256)
    p.add_argument("--router_lambda_bal", type=float, default=0.01)
    p.add_argument("--bernoulli_scheduler", type=str, default="loglinear", choices=["loglinear", "linear"])
    p.add_argument("--corruption_mode", type=str, default="independent", choices=["independent", "trajectory"])
    p.add_argument("--without_T", action="store_true", default=False)
    p.add_argument(
        "--t_start_frac",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 0.9],
        help="Routed train variants at t_start = frac*T (K = T-t_start-1). 0.0 = worst case.",
    )
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--split_forward",
        action="store_true",
        help="Attribute routed forward to router vs denoiser (adds one sync per forward)",
    )
    p.add_argument("--sample_steps", type=int, default=10, help="Reverse steps (num_timesteps_sample)")
    p.add_argument("--sample_batch", type=int, default=64)
    p.add_argument("--sample_iters", type=int, default=3)
    p.add_argument("--sample_warmup", type=int, default=1)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_sample", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", type=str, default="", help="Write all rows to this CSV path")
    p.add_argument(
        "--csv_append",
        action="store_true",
        help="Append to --csv (reusing its header) instead of overwriting -- for bash sweeps over "
        "T / router_conv_kernel / router_out_channels; pair with --tag.",
    )
    p.add_argument("--tag", type=str, default="", help="Free-form label copied into every CSV row")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    T = args.num_timesteps

    routed = RoutedDenoiserCNN(
        d_model=args.d_model,
        max_len=args.seq_len,
        num_timesteps=T,
        num_labels=None,
        router_tau=args.router_tau,
        router_k=args.router_k,
        num_cnn_stacks=args.cnn_stacks,
        router_conv_kernel=args.router_conv_kernel,
        router_out_channels=args.router_out_channels,
    ).to(device)
    baseline = DiscreteDenoiserCNN(
        d_model=args.d_model,
        max_len=args.seq_len,
        num_timesteps=T,
        num_labels=None,
        num_cnn_stacks=args.cnn_stacks,
    ).to(device)

    r_total, r_router = _params(routed)
    b_total, _ = _params(baseline)
    print("=" * 78)
    print("SLM non-Markovian compute-overhead ablation")
    print("=" * 78)
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"torch={torch.__version__}  device={device} ({dev_name})")
    print(
        f"B={args.batch_size}  L={args.seq_len}  T={T}  "
        f"cnn_stacks={args.cnn_stacks}  rk={args.router_conv_kernel}  "
        f"C_out={args.router_out_channels}  tau={args.router_tau}  router_k={args.router_k}  "
        f"corruption_mode={args.corruption_mode}  scheduler={args.bernoulli_scheduler}"
    )
    print(
        f"params: routed={r_total:,}  baseline={b_total:,}  "
        f"router-only={r_router:,} ({100.0 * r_router / max(r_total, 1):.4f}% of routed)"
    )
    print(
        "note: baseline DiscreteDenoiserCNN hardcodes CNNModel num_cls=81 while "
        "RoutedDenoiserCNN uses num_labels or 1, so the routed/baseline total-param "
        "delta also carries a cls_embedder difference -- quote the router-only count "
        "for the parameter-overhead claim."
    )

    rows: list[dict] = []

    if not args.skip_train:
        x0 = torch.randint(0, 4, (args.batch_size, args.seq_len), device=device)
        opt_b = torch.optim.AdamW(baseline.parameters(), lr=3e-4, weight_decay=0.01)
        opt_r = torch.optim.AdamW(routed.parameters(), lr=3e-4, weight_decay=0.01)
        baseline.train()
        routed.train()

        probe: _CnnEntryProbe | None = None
        handle = None
        if args.split_forward:
            probe = _CnnEntryProbe(device)
            handle = routed.cnn.register_forward_pre_hook(probe)

        variants: list[TrainVariant] = [
            TrainVariant(
                "baseline", model=baseline, opt=opt_b, routed=False, t_start=T // 2,
                args=args, device=device, probe=None,
            )
        ]
        for frac in args.t_start_frac:
            t_start = min(max(int(round(frac * T)), 0), T - 1)
            variants.append(
                TrainVariant(
                    f"routed@t_start={t_start}", model=routed, opt=opt_r, routed=True,
                    t_start=t_start, args=args, device=device, probe=probe,
                )
            )
        # K=0: same module, router branch skipped -> isolates the router's cost.
        variants.append(
            TrainVariant(
                "routed_K0", model=routed, opt=opt_r, routed=True, t_start=T - 1,
                args=args, device=device, probe=probe,
            )
        )

        # Round-robin so warmup cost is shared, not charged to whoever runs first.
        for it in range(args.warmup + args.iters):
            for v in variants:
                _reset_peak(device)
                rec = v.step(x0)
                if it >= args.warmup:
                    v.samples.append(rec)
                    v.peak_mem_mb = max(v.peak_mem_mb, _peak_mem_mb(device) or 0.0)
        if handle is not None:
            handle.remove()

        trows = [v.summary(x0) for v in variants]
        rows.extend(trows)
        _print_table(
            trows,
            [
                "variant", "K", "ms_views", "ms_forward", "ms_forward_router",
                "ms_forward_denoiser", "ms_loss", "ms_backward", "ms_step", "ms_step_std",
                "steps_per_s", "peak_mem_mb", "views_buffer_mb", "router_frac_pct",
            ],
            f"Training step, mean of {args.iters} iters (ms, device-synced; "
            "router_frac_pct = analytic router MACs / denoiser MACs)",
        )
        base = trows[0]
        k0 = trows[-1]
        worst = max((r for r in trows if r["variant"].startswith("routed@")), key=lambda r: int(r["K"]))
        print(
            f"\noverhead vs baseline: {worst['variant']} step "
            f"{worst['ms_step'] / base['ms_step']:.2f}x  "
            f"(views {worst['ms_views'] - base['ms_views']:+.2f} ms, "
            f"forward {worst['ms_forward'] - base['ms_forward']:+.2f} ms, "
            f"backward {worst['ms_backward'] - base['ms_backward']:+.2f} ms)"
        )
        print(
            f"router alone (routed@K={worst['K']} minus routed_K0): "
            f"{worst['ms_step'] - k0['ms_step']:+.2f} ms/step "
            f"({100.0 * (worst['ms_step'] - k0['ms_step']) / base['ms_step']:+.1f}% of a baseline step)"
        )
        print(
            f"views buffer at K={worst['K']}: {worst.get('views_buffer_mb', float('nan')):.1f} MB "
            f"(fp32, B x (K+1) x L x 4)"
        )

    if not args.skip_sample:
        routed.eval()
        baseline.eval()
        routed.num_timesteps = args.sample_steps
        baseline.num_timesteps = args.sample_steps
        svariants = [
            SampleVariant("baseline", model=baseline, routed=False, history_mode="", args=args, device=device),
            SampleVariant("routed/trajectory", model=routed, routed=True, history_mode="trajectory", args=args, device=device),
            SampleVariant("routed/uniform", model=routed, routed=True, history_mode="uniform", args=args, device=device),
        ]
        for it in range(args.sample_warmup + args.sample_iters):
            for v in svariants:
                _reset_peak(device)
                ms = v.step(it)
                if it >= args.sample_warmup:
                    v.times.append(ms)
                    v.peak_mem_mb = max(v.peak_mem_mb, _peak_mem_mb(device) or 0.0)
        srows = [v.summary() for v in svariants]
        rows.extend(srows)
        _print_table(
            srows,
            ["variant", "K", "ms_batch", "ms_batch_std", "ms_per_step", "ms_per_seq", "seq_per_s", "peak_mem_mb"],
            f"Sampling: {args.sample_steps} steps, batch {args.sample_batch}, 1 NFE/step, "
            f"mean of {args.sample_iters} runs",
        )
        b = srows[0]
        for r in srows[1:]:
            print(f"  {r['variant']}: {r['ms_batch'] / b['ms_batch']:.2f}x baseline wall-clock")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "tag": args.tag,
            "device": str(device),
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "num_timesteps": T,
            "cnn_stacks": args.cnn_stacks,
            "router_conv_kernel": args.router_conv_kernel,
            "router_out_channels": args.router_out_channels,
            "sample_steps": args.sample_steps,
            "sample_batch": args.sample_batch,
            "params_routed": r_total,
            "params_baseline": b_total,
            "params_router_only": r_router,
        }
        append = args.csv_append and path.exists() and path.stat().st_size > 0
        if append:
            with open(path, newline="", encoding="utf-8") as f:
                fields = next(csv.reader(f))
        else:
            fields = list(cfg.keys())
            for r in rows:
                for k in r:
                    if k not in fields:
                        fields.append(k)
        with open(path, "a" if append else "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not append:
                w.writeheader()
            for r in rows:
                w.writerow({**cfg, **r})
        print(f"\n{'appended' if append else 'wrote'} {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
