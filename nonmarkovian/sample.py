"""Iterative sampling for the routed model.

Mirrors ``sample_simple.sample_sequences_simple`` (same SLM ``new_diff`` reverse
process on a simplex ``x_t``); the only routed-specific difference is the
denoising step, which receives a ``views`` tensor ``[B, T, L, 4]`` built from
the running reverse-process trajectory itself (see ``history_mode``)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.forward import cosine_alpha_schedule, corrupt_sequence_bernoulli, sample_all_views_bernoulli
from nonmarkovian.model import RoutedDenoiserCNN
from nonmarkovian.vocab import IDX_TO_TOKEN


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


def _simplex_to_tokens(x_t: torch.Tensor, undecided: int = 4) -> torch.Tensor:
    """``x_t`` ``[B, L, C]`` simplex -> ``[B, L]`` uint8 ids, ``undecided`` where |support| != 1.

    A position counts as *committed* only once its active set has collapsed to a single class --
    the SLM/new_diff analogue of an MDLM position being unmasked.
    """
    active = x_t > 0
    tok = active.float().argmax(dim=-1)
    return torch.where(active.sum(dim=-1) == 1, tok, torch.full_like(tok, undecided)).to(
        "cpu", torch.uint8
    )


def _simplex_to_bitmask(x_t: torch.Tensor) -> torch.Tensor:
    """``x_t`` ``[B, L, C]`` simplex -> ``[B, L]`` uint8 bitmask of the still-active classes.

    Bit ``c`` is set when class ``c`` is in the position's shortlist. Keeping the whole active set
    (not just "has it collapsed?") is what lets us ask whether the base finally chosen was ever
    *excluded* at an earlier step.
    """
    active = (x_t > 0).to(torch.uint8)
    weights = torch.tensor([1, 2, 4, 8], dtype=torch.uint8, device=x_t.device)[: x_t.shape[-1]]
    return (active * weights).sum(dim=-1).to("cpu", torch.uint8)


def _expected_nums(t: torch.Tensor, *, num_classes: int = 4, scheduler: str = "loglinear") -> torch.Tensor:
    if scheduler == "loglinear":
        return torch.clamp(torch.exp(torch.log(torch.tensor(float(num_classes), device=t.device)) * t), min=1.0)
    if scheduler == "linear":
        return torch.clamp(float(num_classes) * t, min=1.0)
    raise ValueError(f"Unknown Bernoulli scheduler: {scheduler}")


def _init_views_buffer(
    x_t: torch.Tensor,
    num_timesteps: int,
    *,
    hat_x0_ids: torch.Tensor | None = None,
    history_mode: str,
    bernoulli_scheduler: str,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Create the initial ``views`` buffer ``[B, T, L, C]`` for reverse sampling.

    ``history_mode`` options:

    - ``"trajectory"`` *(default)*: start from a uniform simplex ``1/C``
      everywhere. As the reverse process progresses we overwrite slot
      ``t_start`` with the current ``x_t`` at each step, so slots at indices
      ``> t_start`` accumulate the actual past trajectory (noisier states the
      process already visited). Slots at indices ``< t_start`` remain uniform
      because they correspond to less-noisy states not yet reached.
    - ``"uniform"``: keep all non-``t_start`` slots at ``1/C`` for the whole
      reverse process. Matches the ``val_no_history=True`` validation setup
      exactly (useful as a reference / when you want to ignore history).
    - ``"bernoulli_hat"``: legacy mode -- fill non-current slots with
      Bernoulli corruptions of the running ``hat_x_0`` argmax. Out of
      distribution at step 1 (``hat_x_0`` starts at zeros).
    """
    B, L, C = x_t.shape
    T = int(num_timesteps)
    if history_mode in ("trajectory", "uniform"):
        return x_t.new_full((B, T, L, C), 1.0 / float(C))
    if history_mode == "bernoulli_hat":
        if hat_x0_ids is None:
            raise ValueError("bernoulli_hat requires hat_x0_ids")
        return sample_all_views_bernoulli(
            hat_x0_ids,
            T,
            scheduler=bernoulli_scheduler,
            generator=generator,
        )
    raise ValueError(f"Unknown history_mode: {history_mode!r}")


def _refresh_views_buffer(
    views_buffer: torch.Tensor,
    hat_x0_ids: torch.Tensor,
    *,
    history_mode: str,
    bernoulli_scheduler: str,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Per-step refresh of non-current slots (only ``bernoulli_hat`` needs this;
    ``trajectory`` and ``uniform`` are updated in place by the caller)."""
    if history_mode == "bernoulli_hat":
        T = int(views_buffer.shape[1])
        return sample_all_views_bernoulli(
            hat_x0_ids,
            T,
            scheduler=bernoulli_scheduler,
            generator=generator,
        )
    return views_buffer


@torch.no_grad()
def sample_sequences(
    model: RoutedDenoiser | RoutedDenoiserCNN,
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
    history_mode: str = "trajectory",
    corruption_mode: str = "trajectory",
    return_trajectory: bool = False,
    support_constraint: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SLM-style ``new_diff`` reverse sampling with routed (history-aware) denoiser.

    With the default ``history_mode="trajectory"`` the model's ``views`` tensor
    accumulates the reverse-process states themselves: at step ``i`` the
    current simplex ``x_t`` is written into slot ``t_start = num_steps - i``,
    so slots at indices ``> t_start`` hold the simplex inputs used by earlier
    (noisier) denoising steps -- genuine past history. Slots at indices
    ``< t_start`` stay at the uniform simplex ``1/C`` because those
    less-noisy states haven't been reached yet. See
    :func:`_init_views_buffer` for the other modes.

    ``return_trajectory=True`` also returns three trajectories (uint8, CPU) for the
    "did it change its mind?" experiment (see ``mind_change_slm``):

    * ``states`` ``[B, T + 2, L]`` -- the simplex collapsed to token ids, with ``4`` marking a
      position whose active set still has more than one member (i.e. not yet committed). Frame 0
      is the uniform prior (everything undecided), frame ``i`` is ``x_t`` after step ``i``, and the
      last frame is the returned argmax sequence.
    * ``beliefs`` ``[B, T, L]`` -- ``argmax`` of the model's clean-sequence logits at each step,
      read **before** the support mask, i.e. what the model would say regardless of what the
      shortlist permits.
    * ``supports`` ``[B, T + 2, L]`` -- uint8 bitmask of each position's active set (bit ``c`` set
      when class ``c`` is still a candidate), so one can ask whether the base finally chosen was
      ever ruled *out* along the way, not merely whether the committed base changed.
    * ``constrained`` ``[B, T, L]`` -- ``argmax`` of the **support-masked** distribution at each
      step: the base the step would settle on if the shortlist never re-expanded. This is the
      counterfactual answer for a position whose eventual base had been ruled out.

    Frame alignment: ``beliefs[i]`` and ``constrained[i]`` are computed from ``states[i]`` /
    ``supports[i]`` (the state entering step ``i + 1``).

    ``support_constraint=False`` lifts the ``logits.masked_fill(~support_mask, -inf)`` step, which
    otherwise pins an already-collapsed position to its current base.
    """
    model.eval()
    num_steps = int(alphas_sample.shape[0])
    vocab = 4
    #print(num_steps)
    T = num_steps
    #corruption_mode = "trajectory"
    print(corruption_mode,"corruption_mode")

    use_cfg = float(guidance_scale) != 0.0 and labels is not None
    null_lab: torch.Tensor | None = None
    if use_cfg:
        num_cls_for_null = getattr(model, "num_labels", None)
        if num_cls_for_null is not None and int(num_cls_for_null) > 0:
            null_lab = torch.full_like(labels, int(num_cls_for_null))
        else:
            null_lab = None  # DiT backbone uses None as the null path

    x_t = torch.full((batch, seq_len, vocab), 1.0 / float(vocab), device=device, dtype=torch.float32)
    hat_x0_ids = torch.zeros((batch, seq_len), device=device, dtype=torch.long)
    bernoulli_scheduler = "loglinear"
    views_buffer = _init_views_buffer(
        x_t,
        T,
        hat_x0_ids=hat_x0_ids,
        history_mode=history_mode,
        bernoulli_scheduler=bernoulli_scheduler,
        generator=generator,
    )
    frames: list[torch.Tensor] | None = [] if return_trajectory else None
    pred_frames: list[torch.Tensor] | None = [] if return_trajectory else None
    support_frames: list[torch.Tensor] | None = [] if return_trajectory else None
    constrained_frames: list[torch.Tensor] | None = [] if return_trajectory else None
    if frames is not None:
        frames.append(_simplex_to_tokens(x_t))
        support_frames.append(_simplex_to_bitmask(x_t))
    print(bernoulli_scheduler,"bernoulli_scheduler")
    threshold = 6
    print(threshold,"threshold")
    print(use_cfg,"use_cfg")
    for i in range(1, num_steps + 1):
        t = torch.full((batch, 1), 1.0 - float(i - 1) / float(num_steps), device=device, dtype=torch.float32)
        t_start = num_steps - i

        if history_mode == "uniform":
            # True no-history: reset all slots to 1/C at every step so the model
            # only ever sees the current x_t and never accumulates past states.
            views_buffer.fill_(1.0 / float(vocab))
        views_buffer[:, t_start] = x_t
        if use_cfg:
            logits_c, _pi, _h, _lb, seq_in = model(
                views_buffer, t_start, labels=labels, t_cond=float(t[0, 0].item())
            )
            logits_u, _, _, _, _ = model(
                views_buffer, t_start, labels=null_lab, t_cond=float(t[0, 0].item())
            )
            logits = (1.0 + guidance_scale) * logits_c - guidance_scale * logits_u
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        else:
            logits, _pi, _h, _lb, seq_in = model(
                views_buffer, t_start, labels=labels, t_cond=float(t[0, 0].item())
            )

        if pred_frames is not None:
            # Unconstrained belief about x0 at this step: before the support mask pins an
            # already-collapsed position to the single base its shortlist still allows.
            pred_frames.append(logits[..., :vocab].argmax(dim=-1).to("cpu", torch.uint8))

        support_mask = (seq_in > 0) if seq_in is not None else (x_t > 0)
        support_mask = (seq_in >0)
        support_mask = (x_t > 0)
        has_any = support_mask.any(dim=-1, keepdim=True)
        support_mask = torch.where(has_any, support_mask, torch.ones_like(support_mask))
        if not support_constraint:
            support_mask = torch.ones_like(support_mask)
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~support_mask, neg_inf)


        model_prob = F.softmax(logits, dim=-1)
        if not torch.allclose(
            model_prob.sum(dim=-1),
            torch.ones((batch, seq_len), device=device, dtype=model_prob.dtype),
            atol=1e-4,
        ):
            model_prob = model_prob / model_prob.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if constrained_frames is not None:
            # argmax *within the current shortlist* (model_prob is already support-masked): the
            # base this step would settle on if the shortlist never re-expanded.
            constrained_frames.append(model_prob.argmax(dim=-1).to("cpu", torch.uint8))

        hat_x0_ids = model_prob.argmax(dim=-1).clamp(max=vocab - 1)
        one_hot = F.one_hot(hat_x0_ids, num_classes=vocab).to(dtype=torch.bool)

        t3 = t.unsqueeze(-1)
        nominator = _expected_nums(t3 - 1.0 / float(num_steps), scheduler=bernoulli_scheduler) - 1.0
        denominator = torch.clamp(_expected_nums(t3, scheduler=bernoulli_scheduler) - 1.0, min=1e-8)
        weight = torch.clamp(nominator / denominator, min=0.0, max=1.0)
        one_hot_f = one_hot.to(dtype=weight.dtype)
        predicted = torch.clamp(model_prob + weight * (1.0 - model_prob), min=0.0, max=1.0)
        #print(predicted, i,one_hot_f,weight,"predicted")
        # SLM's reverse step keeps only channels that were active in the input
        # simplex. The routed CNN mixes the current view with history, so the
        # effective "input simplex" is ``seq_in``, not the raw ``x_t``. Fall
        # back to ``x_t > 0`` when the backbone doesn't expose ``seq_in``
        # (DiT, or any model returning None).
        eps = 1e-4
        
        # ═══ OLD reverse step (same `predicted` noise for both modes; markov/nonmarkov differ
        #     ONLY by the `& (x_t>0)` constraint). Commented out for A/B testing — uncomment this
        #     block and comment the NEW block below to revert. ══════════════════════════════════
        # if corruption_mode == "independent":
        #     support_mask = (x_t > 0)
        #     if i > threshold*num_steps // 10 :
        #         #print(i,"i")
        #         sample_pred = _sample_bernoulli(predicted, generator=generator)
        #     else:
        #         sample_pred = _sample_bernoulli(predicted, generator=generator) & support_mask
        #     #print(sample_pred, i,"independent")
        # else:
        #     support_mask = (x_t > 0)
        #     sample_pred = _sample_bernoulli(predicted, generator=generator) & support_mask
        # #print(support_mask.shape, support_mask.sum(),x_t, t_start)
        # sample_pred_sum = sample_pred.sum(dim=-1, keepdim=True)
        # fallback = F.one_hot(predicted.argmax(dim=-1), num_classes=vocab).to(dtype=torch.bool)
        # sample_pred = torch.where(sample_pred_sum > 0, sample_pred, fallback)
        # x_t = sample_pred.to(dtype=torch.float32)
        # x_t = x_t / x_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # ═══ NEW: training-matched reverse noise (parallel to sample_mdlm) ══════════════════════
        # The noise added is DIFFERENT per mode so intermediate x_t matches each model's training
        # corruption (correct per-level marginal), instead of sharing one `predicted` + constraint:
        #   * Markov / early steps: schedule-calibrated `predicted` intersected with the current
        #     support -> the active set can only shrink (matches the monotone `trajectory` forward
        #     corruption the Markov model was trained on).
        #   * Non-Markov (`independent`), past the threshold: re-noise the predicted clean sequence
        #     `hat_x0` with the ACTUAL forward Bernoulli corruption at the next (less-noisy) level.
        #     This reproduces the i.i.d. views the independent model was trained on and lets dropped
        #     classes re-activate (support can grow again).
        if corruption_mode == "independent" and i > threshold * num_steps // 10:
            #t_next = torch.clamp(t - 1.0 / float(num_steps), min=1.0 / float(num_steps))
            #x_t = corrupt_sequence_bernoulli(
            #    hat_x0_ids, t_next, scheduler=bernoulli_scheduler, generator=generator
            #)
            t3 = t.unsqueeze(-1)
            nominator = _expected_nums(t3 - 1.0 / float(num_steps), scheduler=bernoulli_scheduler) - 1.0
            denominator = torch.clamp(_expected_nums(t3, scheduler=bernoulli_scheduler) - 1.0, min=1e-8)
            weight = torch.clamp(nominator, min=0.0, max=1.0)
            one_hot_f = one_hot.to(dtype=weight.dtype)
            predicted = torch.clamp(model_prob + weight * (1.0 - model_prob), min=0.0, max=1.0)
            
            sample_pred = _sample_bernoulli(predicted, generator=generator) 
            sample_pred_sum = sample_pred.sum(dim=-1, keepdim=True)
            fallback = F.one_hot(predicted.argmax(dim=-1), num_classes=vocab).to(dtype=torch.bool)
            sample_pred = torch.where(sample_pred_sum > 0, sample_pred, fallback)
            x_t = sample_pred.to(dtype=torch.float32)
            x_t = x_t / x_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            support_mask = (x_t > 0)
            sample_pred = _sample_bernoulli(predicted, generator=generator) & support_mask
            sample_pred_sum = sample_pred.sum(dim=-1, keepdim=True)
            fallback = F.one_hot(predicted.argmax(dim=-1), num_classes=vocab).to(dtype=torch.bool)
            sample_pred = torch.where(sample_pred_sum > 0, sample_pred, fallback)
            x_t = sample_pred.to(dtype=torch.float32)
            x_t = x_t / x_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if frames is not None:
            frames.append(_simplex_to_tokens(x_t))
            support_frames.append(_simplex_to_bitmask(x_t))

    t_last = torch.full((batch, 1), 1.0 / float(num_steps), device=device, dtype=torch.float32)
    views_buffer = _refresh_views_buffer(
        views_buffer,
        hat_x0_ids,
        history_mode=history_mode,
        bernoulli_scheduler=bernoulli_scheduler,
        generator=generator,
    )
    if history_mode == "uniform":
        views_buffer.fill_(1.0 / float(vocab))
    views_buffer[:, 0] = x_t
    if use_cfg:
        logits_c, _pi, _h, _lb, _seq_in = model(
            views_buffer, 0, labels=labels, t_cond=float(t_last[0, 0].item())
        )
        logits_u, _, _, _, _ = model(
            views_buffer, 0, labels=null_lab, t_cond=float(t_last[0, 0].item())
        )
        logits_last = (1.0 + guidance_scale) * logits_c - guidance_scale * logits_u
        logits_last = logits_last - torch.logsumexp(logits_last, dim=-1, keepdim=True)
    else:
        logits_last, _pi, _h, _lb, _seq_in = model(
            views_buffer, 0, labels=labels, t_cond=float(t_last[0, 0].item())
        )
    final = logits_last.argmax(dim=-1).clamp(max=3)
    if frames is not None:
        frames.append(final.to("cpu", torch.uint8))
        support_frames.append(
            (torch.ones_like(final, dtype=torch.uint8) << final.to(torch.uint8)).cpu()
        )
        return (
            final,
            torch.stack(frames, dim=1),
            torch.stack(pred_frames, dim=1),
            torch.stack(support_frames, dim=1),
            torch.stack(constrained_frames, dim=1),
        )
    return final


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
    p.add_argument(
        "--router_topk",
        type=int,
        default=0,
        help="Deprecated; Boltzmann router ignores this (kept for old configs)",
    )
    p.add_argument("--device", type=str, default="auto", help='"auto", "cpu", or "cuda"')
    p.add_argument("--out", type=str, default="samples.txt")
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
        help="SLM-style Bernoulli scheduler for routed new_diff sampling.",
    )
    p.add_argument(
        "--corruption_mode",
        type=str,
        default=None,
        choices=("trajectory", "independent"),
        help=(
            "Forward-corruption mode the model was trained with. Defaults to the value "
            "stored in the checkpoint (cfg['corruption_mode']); pass explicitly to override. "
            "'trajectory' (Markovian, monotone: once a class is dropped it stays dropped) "
            "keeps the support-mask constraint in reverse sampling. 'independent' "
            "(non-Markovian) drops the constraint so dropped classes can re-activate."
        ),
    )
    p.add_argument(
        "--history_mode",
        type=str,
        default="trajectory",
        choices=("trajectory", "uniform", "bernoulli_hat"),
        help=(
            "How to fill non-current view slots during reverse sampling. "
            "'trajectory' (default) accumulates the running reverse-process simplices "
            "themselves: at step i the current x_t is written into slot t_start, so slots at "
            "indices > t_start hold past (noisier) x_t states -- the actual history. "
            "'uniform' keeps non-current slots at 1/C (matches --val_no_history=True validation). "
            "'bernoulli_hat' uses Bernoulli corruptions of the running hat_x0 (legacy)."
        ),
    )
    args = p.parse_args()

    device = resolve_device_arg(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})
    num_timesteps_train = int(cfg.get("num_timesteps", 32))
    num_timesteps_sample = int(cfg.get("num_timesteps_sample", num_timesteps_train))
    if args.num_timesteps_sample > 0:
        num_timesteps_sample = int(args.num_timesteps_sample)
    max_len = int(cfg.get("max_len", 500))
    seq_len = min(args.seq_len, max_len)
    num_classes = int(cfg.get("num_classes", 0))
    if args.corruption_mode is None:
        args.corruption_mode = str(cfg.get("corruption_mode", "trajectory"))
        print(f"Using corruption_mode={args.corruption_mode} (from checkpoint)")
    else:
        print(f"Using corruption_mode={args.corruption_mode} (from CLI)")

    alphas_sample = ckpt.get("alphas_sample")
    if alphas_sample is None:
        alphas_sample = cosine_alpha_schedule(num_timesteps_sample, device=device)
    else:
        alphas_sample = alphas_sample.to(device)
        if alphas_sample.shape[0] != num_timesteps_sample:
            alphas_sample = cosine_alpha_schedule(num_timesteps_sample, device=device)

    router_tau = float(cfg.get("router_tau", 1.0))
    router_k = int(cfg.get("router_k", 1))
    backbone = str(cfg.get("backbone", "dit")).lower()
    cnn_stacks = int(cfg.get("cnn_stacks", 4))
    dec_layers_total = int(cfg.get("dec_layers", 6)) + int(cfg.get("enc_layers", 0))
    if backbone == "cnn":
        model = RoutedDenoiserCNN(
            d_model=int(cfg.get("d_model", 256)),
            max_len=max_len,
            num_timesteps=num_timesteps_sample,
            num_labels=num_classes if num_classes > 0 else None,
            router_tau=router_tau,
            router_k=router_k,
            num_cnn_stacks=cnn_stacks,
            router_conv_kernel=int(cfg.get("router_conv_kernel", 3)),
            router_out_channels=int(cfg.get("router_out_channels", 128)),
        ).to(device)
    else:
        cond_dim_raw = cfg.get("cond_dim", 0)
        cond_dim = int(cond_dim_raw) if cond_dim_raw else None
        if cond_dim == 0:
            cond_dim = None
        time_freq_dim = int(cfg.get("time_freq_dim", 256))
        model = RoutedDenoiser(
            d_model=int(cfg.get("d_model", 256)),
            nhead=int(cfg.get("nhead", 8)),
            dec_layers=dec_layers_total,
            dim_ff=int(cfg.get("dim_ff", 1024)),
            dropout=float(cfg.get("dropout", 0.1)),
            max_len=max_len,
            num_timesteps=num_timesteps_train,
            num_labels=num_classes if num_classes > 0 else None,
            cond_dim=cond_dim,
            router_tau=router_tau,
            router_k=router_k,
            time_freq_dim=time_freq_dim,
        ).to(device)
    model.load_state_dict(ckpt["model"])
    model.num_timesteps = num_timesteps_sample
    #print(model.num_timesteps,"model.num_timesteps")
    labels = None
    if num_classes > 0 and args.label >= 0:
        labels = torch.full((args.batch,), args.label, device=device, dtype=torch.long)

    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    x = sample_sequences(
        model,
        alphas_sample,
        args.batch,
        seq_len,
        device,
        num_timesteps_train=num_timesteps_train,
        labels=labels,
        bernoulli_scheduler=args.bernoulli_scheduler,
        generator=gen,
        history_mode=args.history_mode,
        corruption_mode=args.corruption_mode,
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
