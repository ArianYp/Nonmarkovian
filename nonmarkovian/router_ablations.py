"""Inference-time router ablations for ``RoutedDenoiserCNN`` — paper ablations, no model edits.

Both ablations replace exactly one thing: the distribution ``pi`` the router puts over the
candidate history states. In :class:`~nonmarkovian.model.RoutedDenoiserCNN.forward` that
distribution is the only way the router influences anything --

    ctx_mix = (z_cand * pi.view(B, -1, 1, 1)).sum(dim=1)

-- so swapping ``pi`` is a complete and faithful ablation of the routing mechanism. Everything
downstream (the corruption-weighted mix of ``z_t`` and ``ctx_mix``, the STE threshold, the
simplex renormalisation, the CNN) is untouched, and ``pi`` is discarded by every sampler, so
nothing else in the codebase observes the change.

The two ablations
-----------------
``uniform``
    ``pi_k = 1/K`` for all K candidates: the mixed context becomes the plain mean of the history
    states. Isolates *whether the router's choice matters at all* — it keeps the same amount of
    history information flowing into the denoiser but destroys the selection. If this matches the
    full model, the router is not doing useful work and the gain is from history availability.
    Distinct from ``--history_mode uniform``, which replaces the history *states* with 1/C noise
    (no information); here the states are real, only the weighting is flattened.

``top1``
    ``pi = onehot(argmax_k e_k)``: a hard single pick instead of the tau-softened softmax over
    compatibility scores. Isolates *whether the soft mixture matters* — the router still selects,
    but cannot blend. Note the trained model already runs a fairly peaked softmax at small
    ``router_tau``, so this is the tau -> 0 limit of the deployed router, not a different router.

Both are applied by patching the bound ``_router_forward`` on a *model instance*, so ``model.py``
stays untouched and an unpatched instance is bit-identical to before. The patch is a no-op while
``model.training`` is True (``top1``'s one-hot has no usable gradient and the load-balance loss
assumes the real softmax), so this is strictly an evaluation tool.

Usage::

    from nonmarkovian.router_ablations import apply_router_ablation
    apply_router_ablation(model, "uniform")   # after load_state_dict + model.eval()

or via ``eval_checkpoint.py --router_ablation {none,uniform,top1}`` (Bernoulli/SLM checkpoints) or
``eval_checkpoint_mdlm.py --router_ablation {none,uniform,top1}`` (MDLM checkpoints). Both share
this module: the routed MDLM model is the same ``RoutedDenoiserCNN``, so the patch is identical.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

ROUTER_ABLATIONS = ("none", "uniform", "top1")


def _pi_uniform(e: torch.Tensor) -> torch.Tensor:
    """``[B, K]`` scores -> flat ``1/K`` weights, ignoring the scores entirely."""
    return e.new_full(e.shape, 1.0 / float(e.shape[-1]))


def _pi_top1(e: torch.Tensor) -> torch.Tensor:
    """``[B, K]`` scores -> one-hot on the argmax (ties go to the lowest index, as torch does)."""
    return F.one_hot(e.argmax(dim=-1), num_classes=int(e.shape[-1])).to(dtype=e.dtype)


_PI_FNS = {"uniform": _pi_uniform, "top1": _pi_top1}


def apply_router_ablation(model: torch.nn.Module, mode: str) -> str:
    """Patch ``model._router_forward`` in place so evaluation uses the ablated ``pi``.

    Returns the normalised mode string. ``"none"`` (or empty) leaves the model alone. Calling
    this twice on one instance is refused rather than silently stacking patches.
    """
    mode = (mode or "none").strip().lower()
    if mode not in ROUTER_ABLATIONS:
        raise SystemExit(
            f"--router_ablation must be one of {ROUTER_ABLATIONS}, got {mode!r}"
        )
    if mode == "none":
        return mode
    if not hasattr(model, "_router_forward"):
        raise SystemExit(
            f"{type(model).__name__} has no _router_forward; router ablations only apply to the "
            "routed model (a simple_discrete checkpoint has no router to ablate)."
        )
    prior = getattr(model, "_router_ablation", "none")
    if prior != "none":
        raise SystemExit(f"router ablation {prior!r} is already applied to this model instance.")

    original = model._router_forward
    pi_fn = _PI_FNS[mode]

    def _patched(e: torch.Tensor):
        # Training keeps the real router: one-hot kills the gradient and _load_balance_loss
        # expects the true softmax. These ablations are eval-only by construction.
        if model.training:
            return original(e)
        if e.ndim != 2 or e.shape[-1] == 0:
            return original(e)
        pi = pi_fn(e)
        # pi_soft stays the *true* softmax so any router diagnostics still report what the
        # unablated router would have done; only the returned pi drives the context mix.
        pi_soft = torch.softmax(e / max(float(model.router_tau), 1e-6), dim=-1)
        return pi, pi_soft, pi

    model._router_forward = _patched
    model._router_ablation = mode
    return mode


def remove_router_ablation(model: torch.nn.Module) -> str:
    """Undo :func:`apply_router_ablation`, restoring the class's own ``_router_forward``."""
    prior = getattr(model, "_router_ablation", "none")
    if prior == "none":
        return "none"
    del model._router_forward          # falls back to the class attribute (the real method)
    model._router_ablation = "none"
    return prior


def _selftest() -> None:
    """Shape/semantics check on the pi functions plus a patch round-trip on a real model."""
    e = torch.tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 0.0]])
    u, t1 = _pi_uniform(e), _pi_top1(e)
    assert torch.allclose(u, torch.full_like(e, 1 / 3)), u
    assert torch.allclose(u.sum(-1), torch.ones(2)) and torch.allclose(t1.sum(-1), torch.ones(2))
    assert t1.argmax(-1).tolist() == [1, 0] and set(t1.flatten().tolist()) == {0.0, 1.0}, t1

    from nonmarkovian.model import RoutedDenoiserCNN

    m = RoutedDenoiserCNN(d_model=32, max_len=16, num_timesteps=8, router_conv_kernel=1)
    m.eval()
    base = m._router_forward(e)[0]
    assert apply_router_ablation(m, "uniform") == "uniform"
    assert torch.allclose(m._router_forward(e)[0], u)
    m.train()
    assert torch.allclose(m._router_forward(e)[0], base), "ablation must be inert in train mode"
    m.eval()
    assert remove_router_ablation(m) == "uniform"
    assert torch.allclose(m._router_forward(e)[0], base), "removal must restore the real router"
    print("[router_ablations] selftest OK")


if __name__ == "__main__":
    _selftest()
