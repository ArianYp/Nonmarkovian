"""Non-Markovian forward corruption: independent absorbing [M] noise per timestep."""

from __future__ import annotations

import torch

from nonmarkovian.vocab import MASK_IDX


def cosine_alpha_schedule(num_steps: int, device: torch.device | None = None) -> torch.Tensor:
    """Monotone α_t in [0, 1] with α_1 <= ... <= α_T (t = 1..T)."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    # t linear in [0,1], α = 0.02 + 0.98 * (1 - cos(π t / 2))  gives smooth increase
    t = torch.linspace(0.0, 1.0, num_steps, device=device)
    alpha = 0.02 + 0.98 * (1.0 - torch.cos(0.5 * torch.pi * t))
    alpha = torch.clamp(alpha, min=0.0, max=1.0)
    for i in range(1, len(alpha)):
        if alpha[i] < alpha[i - 1]:
            alpha[i] = alpha[i - 1]
    return alpha


def map_sample_step_to_train_step(
    t_sample: int,
    num_timesteps_train: int,
    num_timesteps_sample: int,
) -> int:
    """Map reverse-process index ``t_sample`` in ``[0, T_sample-1]`` to a training index in ``[0, T_train-1]``.

    Used when sampling runs fewer (or more) steps than training so the denoiser still receives a timestep
    index consistent with how it was trained (cf. SLM training ``T`` vs ``sampling.steps``).
    """
    nt = max(int(num_timesteps_train), 1)
    ns = max(int(num_timesteps_sample), 1)
    ts = int(t_sample)
    if ns <= 1:
        return nt - 1
    return int(round(ts * (nt - 1) / (ns - 1)))


def corrupt_sequence(x0: torch.Tensor, alpha: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Per-position mask with prob alpha; x0 is [B, L] long in 0..3."""
    if generator is None:
        u = torch.rand(x0.shape, device=x0.device, dtype=torch.float32)
    else:
        u = torch.rand(x0.shape, device=x0.device, dtype=torch.float32, generator=generator)
    mask = u < float(alpha)
    out = x0.clone()
    out[mask] = MASK_IDX
    return out


def corrupt_sequence_bernoulli(
    x0: torch.Tensor,
    t: torch.Tensor,
    *,
    num_classes: int = 4,
    scheduler: str = "loglinear",
    generator: torch.Generator | None = None,
    preserve_true_until: float = 0.5,
    high_noise_error_prob: float = 0.15,
    slm_denominator: bool = False,
) -> torch.Tensor:
    """SLM-style Bernoulli corruption on simplex inputs.

    Args:
        x0: ``[B, L]`` token ids in ``0..num_classes-1``.
        t: ``[B, 1]`` normalized timesteps in ``(0, 1]``.
        scheduler: ``loglinear`` or ``linear`` for expected active categories.
        preserve_true_until: for ``t >= preserve_true_until`` (near-clean region),
            always preserve the ground-truth class in ``x_t``.
        high_noise_error_prob: for ``t < preserve_true_until`` (noisy region), probability of
            *not* forcing the ground-truth class to be present.
        slm_denominator: divide ``(E[nums]-1)`` by ``num_classes`` (exact
            ``SLM/slm.py:get_xt_bernoulli`` parity) instead of ``num_classes-1``.
    Returns:
        ``x_t`` probabilities, shape ``[B, L, num_classes]``.
    """
    if scheduler == "loglinear":
        expect_nums = torch.clamp(torch.exp(torch.log(torch.tensor(float(num_classes), device=x0.device)) * t), min=1.0)
    elif scheduler == "linear":
        expect_nums = torch.clamp(float(num_classes) * t, min=1.0)
    else:
        raise ValueError(f"Unknown Bernoulli scheduler: {scheduler}")

    #one_hot = torch.nn.functional.one_hot(x0.long().clamp(min=0, max=num_classes - 1), num_classes=num_classes).to(
    #    dtype=torch.float32
    #)
    one_hot = torch.nn.functional.one_hot(x0.long(), num_classes=num_classes).to(
        dtype=torch.float32
    )
    denom = float(num_classes) if slm_denominator else float(max(num_classes - 1, 1))
    bernoulli_param = (expect_nums - 1.0) / denom
    bernoulli_param = torch.clamp(bernoulli_param, min=0.0, max=1.0).unsqueeze(-1).expand_as(one_hot)
    if generator is None:
        u = torch.rand(one_hot.shape, device=x0.device, dtype=torch.float32)
    else:
        u = torch.rand(one_hot.shape, device=x0.device, dtype=torch.float32, generator=generator)
    samples = (u < bernoulli_param).to(dtype=torch.float32)

    # Near-clean timesteps: always preserve true class.
    # Noisy timesteps: allow dropping the true class with given probability.
    '''
    high_noise = (t < float(preserve_true_until)).unsqueeze(-1)  # [B, 1, 1]
    keep_prob_high_noise = 1.0 - float(high_noise_error_prob)
    if generator is None:
        keep_u = torch.rand((x0.shape[0], x0.shape[1], 1), device=x0.device, dtype=torch.float32)
    else:
        keep_u = torch.rand(
            (x0.shape[0], x0.shape[1], 1), device=x0.device, dtype=torch.float32, generator=generator
        )
    keep_true = torch.where(high_noise, keep_u < keep_prob_high_noise, torch.ones_like(keep_u, dtype=torch.bool))
    keep_true = keep_true.expand_as(one_hot)'''
    x_t = torch.where(one_hot > 0, one_hot, samples)
    #x_t = torch.where((one_hot > 0) & keep_true, one_hot, samples)
    #print(x_t.shape)
    x_t = x_t / x_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return x_t


def get_xt_bernoulli(
    x0: torch.Tensor,
    t: torch.Tensor,
    *,
    num_classes: int = 4,
    scheduler: str = "loglinear",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Alias for SLM naming parity."""
    return corrupt_sequence_bernoulli(
        x0,
        t,
        num_classes=num_classes,
        scheduler=scheduler,
        generator=generator,
    )


def sample_all_views(
    x0: torch.Tensor,
    alphas: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample x_1..x_T independently from x_0. Returns [B, T, L]."""
    B, L = x0.shape
    T = int(alphas.shape[0])
    views = []
    for t in range(T):
        views.append(corrupt_sequence(x0, float(alphas[t].item()), generator=generator))
    return torch.stack(views, dim=1)


def sample_all_views_bernoulli(
    x0: torch.Tensor,
    num_timesteps: int,
    *,
    t_start: int | None = None,
    scheduler: str = "loglinear",
    generator: torch.Generator | None = None,
    num_classes: int = 4,
    corruption_mode: str = "independent",
    slm_denominator: bool = False,
) -> torch.Tensor:
    """Sample Bernoulli simplex views x_1..x_T from x_0.

    Returns shape ``[B, T, L, 4]`` where each view ``tau`` uses ``t=(tau+1)/T``.
    If ``t_start`` is provided, returns only needed views ``tau in [t_start, T-1]``,
    with shape ``[B, T - t_start, L, 4]``.

    ``slm_denominator`` selects the off-diagonal Bernoulli probability divisor:

    - ``False`` *(default, DNA/enhancer)*: ``(E[nums]-1) / (num_classes-1)`` so the
      simplex is fully uniform at ``t=1``.
    - ``True`` *(SLM ``get_xt_bernoulli`` parity, used by the protein task)*:
      ``(E[nums]-1) / num_classes`` -- matches ``SLM/slm.py:get_xt_bernoulli`` exactly.

    ``corruption_mode`` controls how off-diagonal noise entries are sampled:

    - ``"independent"`` *(default)*: each timestep draws a fresh random tensor,
      so noise at different times is i.i.d. given x_0.

    - ``"trajectory"``: a single random tensor ``u ~ U[0,1]`` is shared across
      all timesteps. An off-diagonal entry activates (becomes 1) at the first tau
      where ``u < p_tau``, and stays 1 for all later times. This enforces the
      **monotone trajectory constraint**: if a position is zero at time t-1 it
      must also be zero at time t, i.e. the support of noise never shrinks.
    """
    T = int(num_timesteps)
    if T < 1:
        raise ValueError("num_timesteps must be >= 1")
    tau_begin = 0 if t_start is None else int(t_start)
    if not (0 <= tau_begin < T):
        raise ValueError(f"t_start must be in [0, {T - 1}]")
    if corruption_mode not in ("independent", "trajectory"):
        raise ValueError(f"corruption_mode must be 'independent' or 'trajectory', got {corruption_mode!r}")

    B, L = x0.shape
    K = T - tau_begin
    device = x0.device

    taus = torch.arange(tau_begin, T, device=device, dtype=torch.float32)
    t_cont = (taus + 1.0) / float(T)  # [K]

    if scheduler == "loglinear":
        expect_nums = torch.clamp(
            torch.exp(torch.log(torch.tensor(float(num_classes), device=device)) * t_cont),
            min=1.0,
        )
    elif scheduler == "linear":
        expect_nums = torch.clamp(float(num_classes) * t_cont, min=1.0)
    else:
        raise ValueError(f"Unknown Bernoulli scheduler: {scheduler}")

    denom = float(num_classes) if slm_denominator else float(max(num_classes - 1, 1))
    bernoulli_param = (expect_nums - 1.0) / denom
    bernoulli_param = torch.clamp(bernoulli_param, min=0.0, max=1.0).view(1, K, 1, 1)  # [1, K, 1, 1]

    one_hot = torch.nn.functional.one_hot(x0.long(), num_classes=num_classes).to(
        dtype=torch.float32
    )  # [B, L, C]
    one_hot_k = one_hot.unsqueeze(1).expand(B, K, L, num_classes)

    if corruption_mode == "independent":
        # Fresh u per (B, tau, L, C) — noise at each timestep is i.i.d. given x_0.
        shape = (B, K, L, num_classes)
        if generator is None:
            u = torch.rand(shape, device=device, dtype=torch.float32)
        else:
            u = torch.rand(shape, device=device, dtype=torch.float32, generator=generator)
    else:
        # "trajectory": one u per (B, L, C) shared across all timesteps.
        # Because p_tau is monotonically increasing, u < p_{tau-1} => u < p_tau,
        # so once a position activates it stays active — the monotone constraint holds.
        shape = (B, 1, L, num_classes)
        if generator is None:
            u = torch.rand(shape, device=device, dtype=torch.float32)
        else:
            u = torch.rand(shape, device=device, dtype=torch.float32, generator=generator)
        # u broadcasts over K dimension against bernoulli_param [1, K, 1, 1]
        #print(u.shape, bernoulli_param.shape)

    samples = (u < bernoulli_param).to(dtype=torch.float32)  # [B, K, L, C]

    x_t = torch.where(one_hot_k > 0, one_hot_k, samples)
    x_t = x_t / x_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return x_t


def transition_from_predicted_x0(
    x0_pred: torch.Tensor,
    alpha_prev: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """q(x_{t-1} | x0): keep nucleotide w.p. 1-α, mask w.p. α."""
    return corrupt_sequence(x0_pred, alpha_prev, generator=generator)
