"""How often does the model *change its mind* during SLM (``new_diff``) inference?

The SLM-backbone twin of ``mind_change_mdlm``: same experiment, same metrics (both import
``mind_change_core``), but on the **Bernoulli / ShortListing** reverse process of ``sample.py`` /
``sample_simple.py`` -- i.e. checkpoints from ``train.py`` (``trainer='routed_discrete'``) and
``train_simple.py`` (``trainer='simple_discrete'``), the ones ``eval_checkpoint.py`` evaluates.

What "committed" means here
---------------------------
MDLM state is a token or ``[M]``. SLM state is a **simplex** ``x_t`` whose non-zero entries are the
classes still in play, so the analogue is:

* ``|support| > 1``  -> the position is still **undecided** (plays the role of ``[M]``);
* ``|support| == 1`` -> the position has **committed** to that base.

A mind change is then the same thing as in the MDLM script: the base a position is committed to
differs from the base it was committed to earlier, anywhere later in the trajectory (undecided
frames in between are bridged).

Where a mind change can come from
---------------------------------
``sample.py``'s reverse step has two regimes, split at ``i > threshold * num_steps // 10``
(``threshold = 6``, so 60% of the way through):

* **before** the split (and always, for ``corruption_mode='trajectory'`` and the simple baseline):
  ``_sample_bernoulli(predicted) & (x_t > 0)`` -- the active set can only **shrink**, so a
  collapsed position is frozen. Every switch count is 0 by construction: that is the control.
* **after** the split, with ``corruption_mode='independent'``: the ``& (x_t > 0)`` intersection is
  dropped, so classes can **re-activate** -- a collapsed position can re-expand to several
  candidates and later collapse onto a *different* base. This re-expansion is the SLM analogue of
  MDLM re-masking, and it is the whole point of the non-Markovian variant.

Note the logits are separately masked to the current support before sampling, which pins a
collapsed position's own prediction to the base it already holds; ``--free_support`` lifts that,
exactly as in the MDLM script.

Reported: switches per position / per sequence, % positions ever changed, net changes, the 4x4
from->to matrix, per-step curves, support re-expansion ("remask") counts and how often a
re-expanded position collapses back to the *same* base, plus the mean support size per step. With
``--score_revisions``, whether the changes improved ``log p(class | sequence)`` under the FBCNN,
against two nulls (see ``mind_change_core.RevisionScorer``).

Usage::

    # non-Markovian, fly brain
    python -m nonmarkovian.mind_change_slm \
        --checkpoint wandb/run-.../files/routed.best_fbd.pt \
        --n_samples 256 --num_timesteps_sample 32 \
        --corruption_mode independent --guidance_scale 0.6 \
        --score_revisions --fbcnn_ckpt fbd.ckpt \
        --out_dir logs/mind_change --tag slm_fb

    # Markovian control (expect 0 switches)
    python -m nonmarkovian.mind_change_slm \
        --checkpoint wandb/run-.../files/routed.best_fbd.pt \
        --corruption_mode trajectory --n_samples 256 --tag slm_fb_markov

Single-GPU / CPU only (no ``torchrun``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.eval_checkpoint import (
    _build_args_namespace,
    _build_loader,
    _build_routed_model,
    _build_simple_model,
    _detect_trainer,
)
from nonmarkovian.forward import cosine_alpha_schedule
from nonmarkovian.mind_change_core import (
    NUC,
    UNDECIDED,
    BeliefVsStateStats,
    MindChangeStats,
    EmbeddingProximity,
    RevisionScorer,
    _pick_excluding,
)
from nonmarkovian.sample import sample_sequences
from nonmarkovian.sample_simple import sample_sequences_simple
from nonmarkovian.validation import _use_conditional_sampling_labels
from nonmarkovian.vocab import IDX_TO_TOKEN


class FinalBaseExclusionStats:
    """Was the base the model finally chose ever *ruled out* earlier in the trajectory?

    Reads the active-set bitmasks: for position ``j`` with final base ``b``, a frame counts as an
    exclusion when bit ``b`` is clear, i.e. the shortlist at that step did not contain ``b`` at all
    (channel ``b`` of the one-hot/simplex was 0). This is strictly stronger than counting committed
    switches: a position can rule the eventual answer out and come back to it without ever having
    *committed* to a different base.

    The first frame (uniform prior, everything active) and the last frame (the returned argmax) can
    never be exclusions by construction, so only the intermediate shortlist states can contribute.
    """

    def __init__(self, seq_len: int) -> None:
        self.L = int(seq_len)
        self.n_seqs = 0
        self.n_positions = 0
        self.n_ever_excluded = 0
        self.n_seqs_with_any = 0
        self.sum_steps_excluded = 0.0
        self.step_excluded: torch.Tensor | None = None
        self.pos_ever_excluded = torch.zeros(self.L, dtype=torch.float64)
        self.hist_steps_excluded = torch.zeros(1, dtype=torch.float64)

    @torch.no_grad()
    def update(self, supports: torch.Tensor, final: torch.Tensor) -> None:
        supports = supports.to("cpu").long()          # [B, F, L] bitmask
        final = final.detach().to("cpu").long()        # [B, L]
        B, F_, L = supports.shape
        if self.step_excluded is None:
            self.step_excluded = torch.zeros(F_, dtype=torch.float64)

        bit = (1 << final).unsqueeze(1)               # [B, 1, L]
        excluded = (supports & bit) == 0              # [B, F, L]
        n_steps = excluded.sum(dim=1)                 # [B, L]
        ever = n_steps > 0

        self.n_seqs += B
        self.n_positions += B * L
        self.n_ever_excluded += int(ever.sum())
        self.n_seqs_with_any += int(ever.any(dim=1).sum())
        self.sum_steps_excluded += float(n_steps.sum())
        self.step_excluded += excluded.sum(dim=(0, 2)).to(torch.float64)
        self.pos_ever_excluded += ever.sum(dim=0).to(torch.float64)
        m = int(n_steps.max())
        if m >= self.hist_steps_excluded.numel():
            self.hist_steps_excluded = torch.cat(
                [self.hist_steps_excluded, torch.zeros(m + 1 - self.hist_steps_excluded.numel(), dtype=torch.float64)]
            )
        self.hist_steps_excluded += torch.bincount(
            n_steps.reshape(-1), minlength=self.hist_steps_excluded.numel()
        ).to(torch.float64)

    def summary(self) -> dict:
        if not self.n_positions:
            return {}
        npos = float(self.n_positions)
        return {
            "n_positions": self.n_positions,
            "n_sequences": self.n_seqs,
            "frac_positions_final_base_ever_excluded": self.n_ever_excluded / npos,
            "n_positions_final_base_ever_excluded": self.n_ever_excluded,
            "frac_sequences_with_any": self.n_seqs_with_any / max(self.n_seqs, 1),
            "mean_steps_excluded_per_position": self.sum_steps_excluded / npos,
            "mean_steps_excluded_given_ever": (
                self.sum_steps_excluded / self.n_ever_excluded if self.n_ever_excluded else 0.0
            ),
            "frac_positions_by_steps_excluded": {
                str(i): float(v) / npos
                for i, v in enumerate(self.hist_steps_excluded.tolist())
                if v > 0
            },
            "frac_positions_excluded_per_step": (self.step_excluded / npos).tolist(),
        }


def build_recovery_variants(
    supports: torch.Tensor,
    constrained: torch.Tensor,
    final: torch.Tensor,
    gen: torch.Generator,
    num_classes: int = 4,
    states: torch.Tensor | None = None,
) -> dict:
    """The four sequence sets the recovery experiment compares, per batch.

    ``actual``            what the model output.
    ``best_surviving``    every recovered position reverted to ``c*`` -- the model's argmax *within*
                          the surviving shortlist at the last frame where the final base was still
                          ruled out, i.e. what the step would have settled on without the corrector.
    ``random_surviving``  every recovered position reverted to a uniform member of that shortlist.
    ``random_positions``  the same *number* of random other positions given a random other base
                          (the calibration null).

    Also returns ``recovered`` ``[B, L]`` and the per-sequence count ``k``.
    """
    supports = supports.to("cpu")
    constrained = constrained.to("cpu")
    final = final.detach().to("cpu").long()
    B, F_, L = supports.shape

    bit = (1 << final).unsqueeze(1)
    excluded = (supports.long() & bit) == 0
    idx = torch.arange(F_).view(1, F_, 1).expand(B, F_, L)
    f_star = torch.where(excluded, idx, torch.full_like(idx, -1)).max(dim=1).values
    recovered = f_star >= 0
    k = recovered.sum(dim=1)

    gather_idx = f_star.clamp(min=0).unsqueeze(1)
    c_idx = gather_idx.clamp(max=constrained.shape[1] - 1)
    c_star = constrained.gather(1, c_idx).squeeze(1).long()
    s_bits = supports.gather(1, gather_idx).squeeze(1)

    shifts = torch.arange(num_classes)
    allowed = ((s_bits.long().unsqueeze(-1) >> shifts) & 1).bool()
    c_rand = _pick_excluding(~allowed.reshape(-1, num_classes), gen, num_classes).reshape(B, L).long()

    r = torch.rand(B, L, generator=gen)
    rand_pos = r.argsort(dim=1).argsort(dim=1) < k.unsqueeze(1)
    excl_final = torch.zeros(B * L, num_classes, dtype=torch.bool)
    excl_final[torch.arange(B * L), final.reshape(-1)] = True
    c_any = _pick_excluding(excl_final, gen, num_classes).reshape(B, L).long()

    out = {
        "actual": final,
        "best_surviving": torch.where(recovered, c_star, final),
        "random_surviving": torch.where(recovered, c_rand, final),
        "random_positions": torch.where(rand_pos, c_any, final),
        "recovered": recovered,
        "k": k,
    }

    if states is not None:
        # Second, independent counterfactual: undo the *revisions*. For every position that ended
        # on a different base than the one it FIRST committed to, put the first commitment back --
        # i.e. what the model would have emitted had it never revised a commitment. Gets its own
        # count-matched random-position null, since this set is a different size from the recoveries.
        st = states.to("cpu")
        first = torch.full((B, L), 255, dtype=torch.uint8)
        for f in range(st.shape[1]):
            cur = st[:, f]
            com = cur != UNDECIDED
            first = torch.where(com & (first == 255), cur, first)
        switched = (first != 255) & (first.long() != final)
        k_sw = switched.sum(dim=1)

        r2 = torch.rand(B, L, generator=gen)
        rand_pos_sw = r2.argsort(dim=1).argsort(dim=1) < k_sw.unsqueeze(1)
        c_any2 = _pick_excluding(excl_final, gen, num_classes).reshape(B, L).long()

        # Same positions, but an arbitrary *third* base -- neither the first commitment nor the
        # final one. Separates "the model revised this position" from "the model revised it to the
        # right base": if the actual output beats this, the specific choice carried information.
        first_safe = torch.where(first == 255, final.to(torch.uint8), first).long()
        excl_two = torch.zeros(B * L, num_classes, dtype=torch.bool)
        ar_bl = torch.arange(B * L)
        excl_two[ar_bl, first_safe.reshape(-1)] = True
        excl_two[ar_bl, final.reshape(-1)] = True
        c_third = _pick_excluding(excl_two, gen, num_classes).reshape(B, L).long()

        out.update(
            {
                "first_commitment": torch.where(switched, first.long(), final),
                "random_positions_switch": torch.where(rand_pos_sw, c_any2, final),
                "random_third_switch": torch.where(switched, c_third, final),
                "switched": switched,
                "k_switch": k_sw,
            }
        )
    return out


class RecoveryFBD:
    """Does re-admitting the ruled-out bases move the *sample distribution* toward the real data?

    Per-sequence scoring with this classifier is not trustworthy: its class head puts real held-out
    fly-brain sequences below chance under their own labels, and embedding proximity (Mahalanobis or
    kNN) ranks randomised real sequences *above* real ones, because noise drags a mean-pooled CNN
    embedding toward the dataset centroid. What the embedding does support is the distributional
    comparison FBD is built on, so this class asks the question at that level:

        FBD(real, actual)  vs  FBD(real, counterfactual)

    for each counterfactual set (all recoveries reverted). **Lower FBD is better**, so
    ``delta = FBD(counterfactual) - FBD(actual) > 0`` means the corrector moved the samples toward
    the real distribution. A paired bootstrap over sequences (the same resampled indices used for
    every variant) gives the interval, and the ``random_positions`` row is the sanity check: damage
    must raise FBD, or the measurement is not working.
    """

    def __init__(self, embed_fn, real_embeddings: torch.Tensor, seed: int = 0, num_classes: int = 4) -> None:
        self.embed_fn = embed_fn
        self.real = real_embeddings.detach().cpu().double().numpy()
        self.C = int(num_classes)
        self.gen = torch.Generator().manual_seed(int(seed) + 8675309)
        self.emb: dict[str, list] = {}
        self.k: list[int] = []
        self.k_switch: list[int] = []

    @torch.no_grad()
    def update(self, supports, constrained, final, labels=None, *, states=None, seq_offset: int = 0) -> None:
        v = build_recovery_variants(supports, constrained, final, self.gen, self.C, states=states)
        self.k.extend(v["k"].tolist())
        names = ["actual", "best_surviving", "random_surviving", "random_positions"]
        if "first_commitment" in v:
            self.k_switch.extend(v["k_switch"].tolist())
            names += ["first_commitment", "random_positions_switch", "random_third_switch"]
        for name in names:
            self.emb.setdefault(name, []).append(self.embed_fn(v[name]).detach().cpu().double())

    def summary(self, n_boot: int = 200) -> dict:
        from nonmarkovian.metrics import frechet_distance_np

        if not self.emb:
            return {}
        E = {k: torch.cat(vs).numpy() for k, vs in self.emb.items()}
        n = E["actual"].shape[0]
        fbd = {k: float(frechet_distance_np(self.real, v)) for k, v in E.items()}

        boot_gen = torch.Generator().manual_seed(20260812)
        deltas: dict[str, list] = {k: [] for k in E if k != "actual"}
        for _ in range(int(n_boot)):
            pick = torch.randint(n, (n,), generator=boot_gen).numpy()
            f_act = float(frechet_distance_np(self.real, E["actual"][pick]))
            for k in deltas:
                deltas[k].append(float(frechet_distance_np(self.real, E[k][pick])) - f_act)

        def ci95(vals):
            v = sorted(vals)
            return [v[max(int(0.025 * len(v)) - 1, 0)], v[min(int(0.975 * len(v)), len(v) - 1)]]

        out = {
            "n_sequences": n,
            "n_real_reference": int(self.real.shape[0]),
            "mean_positions_reverted_per_sequence": float(sum(self.k) / max(len(self.k), 1)),
            "mean_positions_reverted_per_sequence_switch": (
                float(sum(self.k_switch) / len(self.k_switch)) if self.k_switch else 0.0
            ),
            "bootstrap": f"{int(n_boot)} paired resamples over sequences",
            "fbd": fbd,
        }
        for k, vals in deltas.items():
            out[f"delta_fbd_{k}"] = {
                "mean": float(sum(vals) / len(vals)),
                "ci95": ci95(vals),
                "frac_boot_positive": float(sum(1 for x in vals if x > 0)) / len(vals),
            }
        return out


class RecoveryCounterfactualScorer:
    """Was re-admitting the ruled-out bases a *good move*, judged per sequence?

    For every position whose final base ``b`` had been excluded from the shortlist, the corrector
    is what put ``b`` back. Without it the position had to settle on a member of the surviving
    shortlist ``S`` (``b`` not in ``S``). So each generated sequence is compared against two
    counterfactuals in which **all** of its recovered positions are reverted at once:

    ``best``    every recovered position -> ``c*``, the model's argmax *within* ``S`` at the last
                frame where ``b`` was still excluded, i.e. what that step would have settled on.
    ``random``  every recovered position -> a uniform member of ``S``.

    and, as the calibration scale, a third in which the *same number* of **random** positions get a
    random different base (``random_positions``). Scores are ``log p(target class | sequence)``.

    One paired measurement per sequence, so the sequence is the unit of analysis and an ordinary
    bootstrap over sequences is the right interval -- no clustering problem.
    """

    def __init__(self, score_fns: dict, seed: int = 0, num_classes: int = 4) -> None:
        """``score_fns``: ``{name: fn(seqs [N, L], labels [N]) -> [N]}``, higher = better. Every
        counterfactual is scored under each named function, so several criteria (class log-prob,
        embedding proximity, ...) can be compared on identical sequences."""
        self.score_fns = dict(score_fns)
        self.C = int(num_classes)
        self.gen = torch.Generator().manual_seed(int(seed) + 8675309)
        self.rows: dict[str, list[tuple]] = {k: [] for k in self.score_fns}

    @staticmethod
    def _bits_to_mask(bits: torch.Tensor, C: int) -> torch.Tensor:
        """uint8 bitmask ``[...]`` -> bool ``[..., C]`` membership."""
        shifts = torch.arange(C, device=bits.device)
        return ((bits.long().unsqueeze(-1) >> shifts) & 1).bool()

    @torch.no_grad()
    def update(
        self,
        supports: torch.Tensor,
        constrained: torch.Tensor,
        final: torch.Tensor,
        labels: torch.Tensor,
        *,
        seq_offset: int = 0,
    ) -> None:
        labels = labels.detach().to("cpu").long().view(-1)
        v = build_recovery_variants(supports, constrained, final, self.gen, self.C)
        final, k = v["actual"], v["k"]
        seq_best, seq_rand, seq_randpos = (
            v["best_surviving"], v["random_surviving"], v["random_positions"],
        )
        B = final.shape[0]

        for name, fn in self.score_fns.items():
            lp = fn(final, labels).detach().cpu().double()
            lp_best = fn(seq_best, labels).detach().cpu().double()
            lp_rand = fn(seq_rand, labels).detach().cpu().double()
            lp_rpos = fn(seq_randpos, labels).detach().cpu().double()
            for i in range(B):
                self.rows[name].append(
                    (
                        seq_offset + i,
                        int(k[i]),
                        float(lp[i]),
                        float(lp_best[i]),
                        float(lp_rand[i]),
                        float(lp_rpos[i]),
                    )
                )

    def summary(self, n_boot: int = 2000) -> dict:
        return {name: self._summary_one(rows, n_boot) for name, rows in self.rows.items()
                if any(r[1] > 0 for r in rows)}

    def _summary_one(self, all_rows: list, n_boot: int = 2000) -> dict:
        rows = [r for r in all_rows if r[1] > 0]     # sequences with at least one recovery
        if not rows:
            return {}
        k = torch.tensor([r[1] for r in rows], dtype=torch.float64)
        lp = torch.tensor([r[2] for r in rows], dtype=torch.float64)
        d = {
            "best_surviving_candidate": lp - torch.tensor([r[3] for r in rows], dtype=torch.float64),
            "random_surviving_candidate": lp - torch.tensor([r[4] for r in rows], dtype=torch.float64),
            "random_positions_same_count": lp - torch.tensor([r[5] for r in rows], dtype=torch.float64),
        }
        boot_gen = torch.Generator().manual_seed(20260812)
        n = len(rows)
        picks = [torch.randint(n, (n,), generator=boot_gen) for _ in range(int(n_boot))]

        def ci95(vals: list[float]) -> list[float]:
            v = torch.tensor(vals, dtype=torch.float64).sort().values
            return [float(v[max(int(0.025 * len(vals)) - 1, 0)]), float(v[min(int(0.975 * len(vals)), len(vals) - 1)])]

        out = {
            "n_sequences_with_recoveries": n,
            "n_sequences_total": len(all_rows),
            "mean_positions_reverted_per_sequence": float(k.mean()),
            "bootstrap": f"{int(n_boot)} resamples over sequences (one paired measurement each)",
        }
        # paired contrast: does reverting the recoveries cost more or less than the same amount of
        # random damage? Shares the `actual` term, so the sequence baseline cancels.
        d["best_vs_null_paired"] = d["best_surviving_candidate"] - d["random_positions_same_count"]
        for name, dv in d.items():
            out[name] = {
                "mean_delta": float(dv.mean()),
                "mean_delta_ci95": ci95([float(dv[p].mean()) for p in picks]),
                "frac_sequences_improved": float((dv > 0).sum()) / n,
                "frac_sequences_improved_ci95": ci95(
                    [float((dv[p] > 0).sum()) / n for p in picks]
                ),
                "median_delta": float(dv.median()),
                "std_delta": float(dv.std(unbiased=True)),
            }
        return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Count how often the SLM/new_diff model revises a committed base during inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--trainer", type=str, default="", choices=("", "routed_discrete", "simple_discrete"))
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=0, help="0 = ckpt max_len.")
    p.add_argument("--num_timesteps_sample", type=int, default=0, help="0 = ckpt value.")
    p.add_argument("--split", type=str, default="val", choices=("val", "test"))
    p.add_argument("--dfm_enhancer", type=str, default="")
    p.add_argument("--dfm_melanoma", dest="dfm_melanoma", action="store_true", default=None)
    p.add_argument("--no_dfm_melanoma", dest="dfm_melanoma", action="store_false")
    p.add_argument(
        "--history_mode", type=str, default="", choices=("", "trajectory", "uniform"),
        help="Routed-only; empty = ckpt value.",
    )
    p.add_argument(
        "--corruption_mode", type=str, default="", choices=("", "independent", "trajectory"),
        # NB: '%%' -- ArgumentDefaultsHelpFormatter %-formats help strings.
        help="Routed-only. 'independent' drops the support intersection past the sampler's "
        "threshold fraction of the steps (see sample.py's `threshold`) "
        "(non-Markovian); 'trajectory' keeps it throughout (Markovian control, 0 switches).",
    )
    p.add_argument("--guidance_scale", type=float, default=0.0)
    p.add_argument(
        "--bernoulli_scheduler", type=str, default="", choices=("", "loglinear", "linear"),
        help="Empty = ckpt value.",
    )
    p.add_argument(
        "--label", type=int, default=-1,
        help="Force one conditioning class; -1 = take labels from the data split.",
    )
    p.add_argument("--unconditional", action="store_true")
    p.add_argument(
        "--free_support", action="store_true",
        help="Routed-only: drop the logit support mask so a collapsed position's own prediction is "
        "not pinned to the base it already holds.",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="logs/mind_change")
    p.add_argument("--tag", type=str, default="", help="Prefix for the output filenames.")
    p.add_argument("--save_trajectories", type=int, default=0)
    p.add_argument(
        "--dump_sequences", type=str, default="",
        help="Write the generated sequences (one ACGT line each) to this path, e.g. for "
        "nonmarkovian.motif_metrics. Conditional checkpoints get labels drawn from --split, so the "
        "class mix matches the real data.",
    )
    p.add_argument("--score_revisions", action="store_true")
    p.add_argument("--fbcnn_ckpt", type=str, default="")
    p.add_argument("--fbcnn_num_cls", type=int, default=0)
    p.add_argument("--fbcnn_stacks", type=int, default=0)
    p.add_argument("--score_max_events", type=int, default=0)
    p.add_argument(
        "--recovery_metric", type=str, default="fbd",
        choices=("fbd", "embedding", "class_logprob", "all"),
        help="Criterion for --score_recoveries. 'fbd' (default) = FBD between the real set and "
        "each counterfactual SET -- distributional, which is the only use of this classifier that "
        "survives a positive control. The per-sequence criteria are kept for reference but both "
        "FAIL that control on fly brain: 'class_logprob' scores real held-out sequences below "
        "chance under their own labels, and 'embedding' (Mahalanobis proximity) ranks randomised "
        "real sequences above real ones. 'all' reports every criterion.",
    )
    p.add_argument("--fbd_boot", type=int, default=200, help="Paired bootstrap draws for the FBD deltas.")
    p.add_argument(
        "--n_ref", type=int, default=2048,
        help="Real sequences used to fit the reference embedding distribution.",
    )
    p.add_argument(
        "--score_recoveries", action="store_true",
        help="Judge the corrector's *recoveries*: for every position whose final base had been "
        "ruled out of the shortlist, revert all of them in one sequence and score it against the "
        "actual output. Needs --fbcnn_ckpt. Reports the best-surviving-candidate counterfactual, "
        "a random-surviving-candidate version, and a count-matched random-position null.",
    )
    p.add_argument("--no_strict_load", action="store_true")
    cli = p.parse_args()

    device = resolve_device_arg(cli.device)
    ckpt_path = Path(cli.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = dict(ckpt.get("args", {}))
    if not cfg:
        raise SystemExit("Checkpoint is missing the 'args' key; cannot reconstruct the model.")
    state = dict(ckpt.get("model") or {})
    if not state:
        raise SystemExit("Checkpoint missing 'model' state_dict.")

    w_phi = state.get("W_phi.weight")
    if isinstance(w_phi, torch.Tensor) and w_phi.ndim == 3:
        cfg["router_out_channels"] = int(w_phi.shape[0])
        cfg["router_conv_kernel"] = int(w_phi.shape[2])

    trainer = cli.trainer.strip().lower() or _detect_trainer(ckpt)
    if trainer not in ("routed_discrete", "simple_discrete"):
        raise SystemExit(f"Unknown trainer type: {trainer!r}")
    if cli.free_support and trainer != "routed_discrete":
        print("[mind-change-slm] warning: --free_support only applies to routed_discrete; ignoring.")

    overrides: dict = {"guidance_scale": float(cli.guidance_scale), "seed": int(cli.seed)}
    if cli.num_timesteps_sample > 0:
        overrides["num_timesteps_sample"] = int(cli.num_timesteps_sample)
    if cli.history_mode:
        overrides["history_mode"] = cli.history_mode
    if cli.corruption_mode:
        overrides["corruption_mode"] = cli.corruption_mode
    if cli.bernoulli_scheduler:
        overrides["bernoulli_scheduler"] = cli.bernoulli_scheduler
    if cli.unconditional:
        overrides["no_labels"] = True
    args = _build_args_namespace(cfg, overrides)

    model = (
        _build_routed_model(cfg, device)
        if trainer == "routed_discrete"
        else _build_simple_model(cfg, device)
    )
    if trainer == "routed_discrete":
        model.num_timesteps = int(args.num_timesteps_sample)
    model.load_state_dict(state, strict=not cli.no_strict_load)
    model.eval()

    T = int(args.num_timesteps_sample)
    seq_len = int(cli.seq_len) if cli.seq_len > 0 else int(getattr(args, "max_len", 500))
    scheduler = str(getattr(args, "bernoulli_scheduler", "loglinear"))
    corruption_mode = str(getattr(args, "corruption_mode", "trajectory"))
    use_labs = _use_conditional_sampling_labels(args)

    alphas_sample = ckpt.get("alphas_sample")
    if alphas_sample is None or alphas_sample.shape[0] != T:
        alphas_sample = cosine_alpha_schedule(T, device=device)
    else:
        alphas_sample = alphas_sample.to(device)

    scorer = None
    rec_scorer = None
    fbd_scorer = None
    if cli.score_revisions or cli.score_recoveries:
        if not cli.fbcnn_ckpt.strip():
            raise SystemExit("--score_revisions/--score_recoveries need --fbcnn_ckpt (e.g. fbd.ckpt).")
        if not use_labs:
            raise SystemExit(
                "Scoring uses log p(target class | sequence), so it needs a conditional "
                "checkpoint sampled with labels (drop --unconditional)."
            )
        from nonmarkovian.fbcnn import load_fbcnn_classifier

        fbcnn = load_fbcnn_classifier(
            cli.fbcnn_ckpt.strip(), device,
            num_cls=int(cli.fbcnn_num_cls or 0), num_cnn_stacks=int(cli.fbcnn_stacks or 0),
        )

        @torch.no_grad()
        def _score_fn(seqs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            s = seqs.to(device)
            t = torch.zeros(s.shape[0], device=device, dtype=torch.float32)
            logits, _emb = fbcnn(s, t, cls=None, return_embedding=True)
            lp = torch.log_softmax(logits.float(), dim=-1)
            return lp.gather(1, labels.view(-1, 1).to(device)).squeeze(1)

        if cli.score_revisions:
            scorer = RevisionScorer(_score_fn, seed=int(cli.seed))

        if cli.score_recoveries:
            rec_fns: dict = {}
            if cli.recovery_metric in ("class_logprob", "all"):
                rec_fns["class_logprob"] = _score_fn
            if cli.recovery_metric in ("fbd", "embedding", "all"):
                from nonmarkovian.metrics import fbcnn_embed_sequences

                @torch.no_grad()
                def _embed(seqs: torch.Tensor) -> torch.Tensor:
                    return fbcnn_embed_sequences(fbcnn, seqs.to(device)).cpu()

                # Reference cloud: real sequences from the eval split, embedded the same way.
                ref_loader = _build_loader(
                    cfg, cli.split, batch_size=256,
                    dfm_root_override=cli.dfm_enhancer, melanoma_override=cli.dfm_melanoma,
                )
                ref_e, ref_l, got = [], [], 0
                for rb in ref_loader:
                    if got >= int(cli.n_ref):
                        break
                    take_r = min(rb["x0"].shape[0], int(cli.n_ref) - got)
                    ref_e.append(_embed(rb["x0"][:take_r].long()))
                    lb = rb.get("label")
                    ref_l.append(
                        lb[:take_r].long() if lb is not None else torch.zeros(take_r, dtype=torch.long)
                    )
                    got += take_r
                ref_e = torch.cat(ref_e)
                ref_l = torch.cat(ref_l)
                print(f"[mind-change-slm] reference embeddings: {tuple(ref_e.shape)} from {got} "
                      f"real {cli.split} sequences")
                if cli.recovery_metric in ("embedding", "all"):
                    rec_fns["embedding_global"] = EmbeddingProximity(_embed, ref_e, ref_l)
                    prox_c = EmbeddingProximity(_embed, ref_e, ref_l, class_conditional=True)
                    if prox_c.n_classes_with_mean:
                        rec_fns["embedding_class"] = prox_c
                if cli.recovery_metric in ("fbd", "all"):
                    fbd_scorer = RecoveryFBD(_embed, ref_e, seed=int(cli.seed))
            rec_scorer = (
                RecoveryCounterfactualScorer(rec_fns, seed=int(cli.seed)) if rec_fns else None
            )

    def _new_label_iter():
        return iter(
            _build_loader(
                cfg, cli.split, batch_size=int(cli.batch),
                dfm_root_override=cli.dfm_enhancer, melanoma_override=cli.dfm_melanoma,
            )
        )

    label_iter = _new_label_iter() if (use_labs and cli.label < 0) else None

    print(
        f"[mind-change-slm] checkpoint={ckpt_path.resolve()} trainer={trainer}\n"
        f"[mind-change-slm] n_samples={cli.n_samples} seq_len={seq_len} steps={T} "
        f"scheduler={scheduler} "
        f"corruption_mode={corruption_mode if trainer == 'routed_discrete' else 'support-intersect'} "
        f"history_mode={getattr(args, 'history_mode', 'n/a')} "
        f"guidance_scale={float(args.guidance_scale)} labels={'yes' if use_labs else 'no'} "
        f"support_constraint={'off (--free_support)' if cli.free_support else 'on'}"
    )

    stats = MindChangeStats(seq_len=seq_len, num_frames=T + 2)
    stats_belief = MindChangeStats(seq_len=seq_len, num_frames=T)
    stats_bvs = BeliefVsStateStats(seq_len=seq_len)
    stats_excl = FinalBaseExclusionStats(seq_len=seq_len)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cli.seed))
    kept_states, kept_beliefs = [], []
    dumped: list[str] = []
    support_curve = [0.0] * (T + 2)
    n_batches = 0

    collected = 0
    while collected < int(cli.n_samples):
        take = min(int(cli.batch), int(cli.n_samples) - collected)
        labels = None
        if use_labs:
            if cli.label >= 0:
                labels = torch.full((take,), int(cli.label), device=device, dtype=torch.long)
            elif label_iter is not None:
                try:
                    batch = next(label_iter)
                except StopIteration:
                    label_iter = _new_label_iter()
                    batch = next(label_iter)
                lab = batch.get("label")
                if lab is not None:
                    lab = lab.to(device)
                    if lab.shape[0] < take:
                        lab = lab.repeat((take + lab.shape[0] - 1) // lab.shape[0])
                    labels = lab[:take]

        if trainer == "routed_discrete":
            x_final, states, beliefs, supports, constrained = sample_sequences(
                model, alphas_sample, take, seq_len, device,
                num_timesteps_train=int(getattr(args, "num_timesteps", T)),
                labels=labels,
                guidance_scale=float(args.guidance_scale),
                bernoulli_scheduler=scheduler,
                generator=gen,
                history_mode=str(getattr(args, "history_mode", "trajectory")),
                corruption_mode=corruption_mode,
                return_trajectory=True,
                support_constraint=not cli.free_support,
            )
        else:
            x_final, states, beliefs, supports, constrained = sample_sequences_simple(
                model, alphas_sample, take, seq_len, device,
                num_timesteps_train=int(getattr(args, "num_timesteps", T)),
                labels=labels,
                guidance_scale=float(args.guidance_scale),
                bernoulli_scheduler=scheduler,
                generator=gen,
                return_trajectory=True,
            )

        stats.update(states)
        stats_belief.update(beliefs, final=x_final.to("cpu", torch.uint8))
        stats_bvs.update(states, beliefs)
        stats_excl.update(supports, x_final)
        if rec_scorer is not None and labels is not None:
            rec_scorer.update(supports, constrained, x_final, labels, seq_offset=collected)
        if fbd_scorer is not None:
            fbd_scorer.update(supports, constrained, x_final, states=states, seq_offset=collected)
        if scorer is not None and labels is not None:
            scorer.update(
                states, x_final, labels,
                max_events=int(cli.score_max_events), seq_offset=collected,
            )
        # mean support size per frame: undecided frames carry >1 candidate, so the *fraction*
        # undecided is the readable summary of how far the shortlist has collapsed.
        for f in range(states.shape[1]):
            support_curve[f] += float((states[:, f] == UNDECIDED).float().mean())
        if cli.save_trajectories > 0 and sum(s.shape[0] for s in kept_states) < cli.save_trajectories:
            need = cli.save_trajectories - sum(s.shape[0] for s in kept_states)
            kept_states.append(states[:need].clone())
            kept_beliefs.append(beliefs[:need].clone())
        if cli.dump_sequences:
            for row in x_final.detach().cpu().tolist():
                dumped.append("".join(IDX_TO_TOKEN[int(t)] for t in row))
        n_batches += 1
        collected += take
        print(f"[mind-change-slm] {collected}/{cli.n_samples} sequences", flush=True)

    summary = {
        "state": stats.summary(),
        "belief": stats_belief.summary(),
        "belief_vs_state": stats_bvs.summary(),
        "final_base_ever_excluded": stats_excl.summary(),
        "slm": {
            "frac_undecided_per_step": [v / max(n_batches, 1) for v in support_curve],
        },
        "config": {
            "checkpoint": str(ckpt_path.resolve()),
            "trainer": trainer,
            "num_timesteps_sample": T,
            "bernoulli_scheduler": scheduler,
            "corruption_mode": corruption_mode if trainer == "routed_discrete" else "support_intersect",
            "history_mode": str(getattr(args, "history_mode", "n/a")),
            "guidance_scale": float(args.guidance_scale),
            "label": int(cli.label),
            "conditional": bool(use_labs),
            "seed": int(cli.seed),
            "split": cli.split,
            "support_constraint": not bool(cli.free_support),
        },
    }
    if scorer is not None:
        summary["revision_quality"] = scorer.summary()
    if rec_scorer is not None:
        summary["recovery_quality"] = rec_scorer.summary()
    if fbd_scorer is not None:
        summary["recovery_fbd"] = fbd_scorer.summary(n_boot=int(cli.fbd_boot))

    if cli.dump_sequences:
        dump_path = Path(cli.dump_sequences)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            f.write("\n".join(dumped) + "\n")
        print(f"[mind-change-slm] wrote {len(dumped)} sequences to {dump_path}")

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (cli.tag + "_") if cli.tag else ""
    json_path = out_dir / f"{tag}mind_change_slm.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / f"{tag}mind_change_slm_per_sequence.csv", "w", encoding="utf-8") as f:
        f.write(
            "seq_idx,state_n_switches,state_n_positions_changed,state_n_switches_adjacent,"
            "state_n_support_reexpansions,state_n_net_changed,belief_n_switches,"
            "belief_n_positions_changed,belief_n_net_changed\n"
        )
        for r, rb in zip(stats.rows, stats_belief.rows):
            f.write(
                f"{r[0]},{r[1]:.0f},{r[2]:.0f},{r[3]:.0f},{r[4]:.0f},{r[5]:.0f},"
                f"{rb[1]:.0f},{rb[2]:.0f},{rb[5]:.0f}\n"
            )
    n = max(stats.n_seqs, 1)
    with open(out_dir / f"{tag}mind_change_slm_per_position.csv", "w", encoding="utf-8") as f:
        f.write(
            "position,state_mean_switches,state_frac_ever_changed,"
            "state_mean_support_reexpansions,state_frac_net_changed,belief_mean_switches\n"
        )
        for j in range(stats.L):
            f.write(
                f"{j},{stats.pos_switches[j] / n:.6f},{stats.pos_ever_switched[j] / n:.6f},"
                f"{stats.pos_remasks[j] / n:.6f},{stats.pos_net_changed[j] / n:.6f},"
                f"{stats_belief.pos_switches[j] / n:.6f}\n"
            )
    if scorer is not None and scorer.rows:
        with open(out_dir / f"{tag}revision_scores.csv", "w", encoding="utf-8") as f:
            f.write("seq_idx,position,from,to,final,delta_revision,delta_random_token,delta_random_position\n")
            for r in scorer.rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.6f},{r[6]:.6f},{r[7]:.6f}\n")
    if rec_scorer is not None:
        for metric, rws in rec_scorer.rows.items():
            if not rws:
                continue
            with open(out_dir / f"{tag}recovery_scores_{metric}.csv", "w", encoding="utf-8") as f:
                f.write(
                    "seq_idx,n_positions_reverted,score_actual,score_best_surviving,"
                    "score_random_surviving,score_random_positions\n"
                )
                for r in rws:
                    f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f}\n")

    if kept_states:
        torch.save(
            {"states": torch.cat(kept_states), "beliefs": torch.cat(kept_beliefs)},
            out_dir / f"{tag}trajectories.pt",
        )

    def _report(name: str, s: dict, *, reexpand: bool) -> None:
        print(f"\n[mind-change-slm] === {name} ===")
        print(f"  sequences                        : {s['n_sequences']}  (L={s['seq_len']}, frames={s['num_frames']})")
        print(f"  mean switches / position         : {s['mean_switches_per_position']:.4f}")
        print(f"  mean switches / sequence         : {s['mean_switches_per_sequence']:.2f}")
        print(f"  positions ever changed           : {s['frac_positions_ever_changed'] * 100:.2f}%")
        print(f"  positions net-changed (1st!=final): {s['frac_positions_net_changed'] * 100:.2f}%")
        print(f"  adjacent-frame flips             : {s['mean_switches_adjacent_per_position']:.4f} / position")
        if reexpand:
            print(f"  support re-expansions / position : {s['mean_remasks_per_position']:.4f}")
            if s.get("remask_refill_events"):
                print(
                    f"  re-expand -> re-collapse events  : {s['remask_refill_events']:.0f}, of which "
                    f"{s['frac_refills_restoring_same_token'] * 100:.1f}% came back as the same base"
                )
        if "frac_positions_belief_final_mismatch" in s:
            print(f"  belief != final base (all steps) : {s['frac_positions_belief_final_mismatch'] * 100:.2f}%")
        print(f"  max switches at one position     : {s['max_switches_at_a_position']}")
        print(f"  mean distinct bases visited      : {s['mean_distinct_tokens_per_position']:.4f}")
        print("  switch-count distribution over positions:")
        for k, v in sorted(s["frac_positions_by_switch_count"].items(), key=lambda kv: int(kv[0]))[:8]:
            print(f"    {k:>3} switches: {v * 100:6.2f}%")
        print("  from -> to switch matrix (counts):")
        print("        " + "  ".join(f"{c:>10}" for c in NUC))
        for c in NUC:
            row = s["switch_matrix_counts"][c]
            print(f"    {c} " + "  ".join(f"{row[d]:>10.0f}" for d in NUC))

    _report("STATE trajectory (shortlist collapsed to a base)", summary["state"], reexpand=True)
    _report("BELIEF trajectory (argmax logits, before the support mask)", summary["belief"], reexpand=False)

    ex = summary["final_base_ever_excluded"]
    if ex:
        print("\n[mind-change-slm] === FINAL BASE RULED OUT EARLIER? ===")
        print("  (channel is 1 in the final one-hot, but was 0 in at least one earlier step)")
        print(f"  positions                        : {ex['frac_positions_final_base_ever_excluded'] * 100:.2f}% "
              f"({ex['n_positions_final_base_ever_excluded']} of {ex['n_positions']})")
        print(f"  sequences with >= 1 such position: {ex['frac_sequences_with_any'] * 100:.2f}%")
        print(f"  mean steps excluded / position   : {ex['mean_steps_excluded_per_position']:.4f} "
              f"(given ever excluded: {ex['mean_steps_excluded_given_ever']:.2f})")
        print("  distribution over positions:")
        for k, v in sorted(ex["frac_positions_by_steps_excluded"].items(), key=lambda kv: int(kv[0]))[:8]:
            print(f"    excluded at {k:>3} steps: {v * 100:6.2f}%")
        print(f"  per step: {[round(v * 100, 2) for v in ex['frac_positions_excluded_per_step']]}")

    rf = summary.get("recovery_fbd") or {}
    if rf:
        print("\n[mind-change-slm] === DID THE RECOVERIES MOVE THE SAMPLES TOWARD REAL DATA? [FBD] ===")
        print(f"  {rf['n_sequences']} generated vs {rf['n_real_reference']} real; "
              f"mean positions reverted / sequence: {rf['mean_positions_reverted_per_sequence']:.1f}")
        print(f"  CI95 = {rf['bootstrap']};  lower FBD = closer to real")
        print(f"    FBD(real, actual)            = {rf['fbd']['actual']:.4f}")
        for k in ("best_surviving", "random_surviving", "random_positions"):
            print(f"    FBD(real, {k:<17}) = {rf['fbd'][k]:.4f}")
        print("  delta = FBD(counterfactual) - FBD(actual);  > 0 means the corrector helped")
        for k, name in (
            ("best_surviving", "vs best surviving candidate (c*)"),
            ("random_surviving", "vs random surviving candidate"),
            ("random_positions", "null: same count, random positions"),
        ):
            b = rf[f"delta_fbd_{k}"]
            lo, hi = b["ci95"]
            print(f"  {name:<40}{b['mean']:+10.4f}  CI95=[{lo:+.4f}, {hi:+.4f}]  "
                  f"(bootstrap draws > 0: {b['frac_boot_positive'] * 100:.0f}%)")
        if "delta_fbd_first_commitment" in rf:
            print("\n[mind-change-slm] === DID THE REVISIONS HELP? "
                  "(revert switched positions to their FIRST commitment) [FBD] ===")
            print(f"  mean positions reverted / sequence: "
                  f"{rf['mean_positions_reverted_per_sequence_switch']:.1f}")
            print(f"    FBD(real, actual)              = {rf['fbd']['actual']:.4f}")
            print(f"    FBD(real, first_commitment)    = {rf['fbd']['first_commitment']:.4f}")
            print(f"    FBD(real, random_positions_sw) = {rf['fbd']['random_positions_switch']:.4f}")
            print(f"    FBD(real, random_third_switch) = {rf['fbd']['random_third_switch']:.4f}")
            for k, name in (
                ("first_commitment", "vs the position's first commitment"),
                ("random_third_switch", "vs a random 3rd base (not 1st, not final)"),
                ("random_positions_switch", "null: same count, random positions"),
            ):
                b = rf[f"delta_fbd_{k}"]
                lo, hi = b["ci95"]
                print(f"  {name:<40}{b['mean']:+10.4f}  CI95=[{lo:+.4f}, {hi:+.4f}]  "
                      f"(bootstrap draws > 0: {b['frac_boot_positive'] * 100:.0f}%)")
            nb2 = rf["delta_fbd_random_positions_switch"]
            fc = rf["delta_fbd_first_commitment"]
            if nb2["ci95"][0] <= 0.0:
                print("  !! the null does not raise FBD, so this comparison cannot be read.")
            elif fc["ci95"][0] > 0:
                print("  -> keeping the first commitment would have been WORSE: the revisions "
                      "improve the sample distribution.")
            elif fc["ci95"][1] < 0:
                print("  -> keeping the first commitment would have been BETTER: the revisions "
                      "are hurting sample quality.")
            else:
                print("  -> the revisions have no resolvable effect on the distribution.")

        nb = rf["delta_fbd_random_positions"]
        if nb["ci95"][0] <= 0.0:
            print("  !! the null does not raise FBD: even deliberate damage is not detected, so this "
                  "measurement cannot judge the corrector. Increase --n_samples (FBD needs enough "
                  "sequences for a stable 128x128 covariance) before reading the rows above.")
        else:
            bs = rf["delta_fbd_best_surviving"]
            if bs["ci95"][0] > 0:
                print("  -> reverting the recoveries makes the sample distribution WORSE: the "
                      "corrector genuinely moves samples toward the real data.")
            elif bs["ci95"][1] < 0:
                print("  -> reverting the recoveries makes the distribution BETTER: the corrector "
                      "is hurting sample quality.")
            else:
                print("  -> no resolvable effect on the sample distribution (the null does register, "
                      "so the measurement itself works).")

    rc_all = summary.get("recovery_quality") or {}
    _METRIC_DESC = {
        "embedding_global": "proximity to the real embedding cloud (Mahalanobis, global mean)",
        "embedding_class": "proximity to real embeddings OF THE SAME CLASS (class mean)",
        "class_logprob": "classifier class head log p(class | seq)  [unreliable on fly brain]",
    }
    for metric, rc in rc_all.items():
        if not rc:
            continue
        print(f"\n[mind-change-slm] === WAS RE-ADMITTING THE RULED-OUT BASES A GOOD MOVE? "
              f"[{metric}] ===")
        print(f"  criterion: {_METRIC_DESC.get(metric, metric)}  (higher = better)")
        print(f"  sequences with >=1 recovery: {rc['n_sequences_with_recoveries']} of "
              f"{rc['n_sequences_total']};  mean positions reverted: "
              f"{rc['mean_positions_reverted_per_sequence']:.1f}")
        print(f"  CI95 = {rc['bootstrap']}")
        print("  delta = score(actual) - score(counterfactual);  > 0 means the corrector helped")
        print(f"  {'':44}{'% seqs improved (CI95)':>26}  {'mean delta (CI95)':>30}")
        for key, name in (
            ("best_surviving_candidate", "vs best surviving candidate (c*)"),
            ("random_surviving_candidate", "vs random surviving candidate"),
            ("random_positions_same_count", "null: same count, random positions"),
            ("best_vs_null_paired", "PAIRED: (vs c*) - (vs null)"),
        ):
            b = rc.get(key)
            if not b:
                continue
            fl, fh = b["frac_sequences_improved_ci95"]
            ml, mh = b["mean_delta_ci95"]
            print(f"  {name:<44}{b['frac_sequences_improved'] * 100:7.2f} [{fl * 100:5.2f}, {fh * 100:5.2f}]"
                  f"  {b['mean_delta']:+11.4f} [{ml:+.4f}, {mh:+.4f}]")
        nb = rc["random_positions_same_count"]
        lo, hi = rc["best_surviving_candidate"]["mean_delta_ci95"]
        if nb["mean_delta_ci95"][0] <= 0.0 <= nb["mean_delta_ci95"][1]:
            print("  !! the null is indistinguishable from zero: this criterion cannot even detect "
                  "the same amount of random damage, so it cannot judge the corrector either.")
        elif lo > 0:
            print("  -> the recovered bases score better than what the sampler would otherwise have "
                  "settled on: the corrector is making real corrections.")
        elif hi < 0:
            print("  -> the recovered bases score *worse* than the surviving alternative.")
        else:
            print("  -> indistinguishable by this criterion (but the null does register, so the "
                  "criterion itself is sensitive).")

    und = summary["slm"]["frac_undecided_per_step"]
    print("\n[mind-change-slm] === SHORTLIST COLLAPSE ===")
    print(f"  fraction of positions still undecided, first -> last step: "
          f"{und[0] * 100:.1f}% -> {und[-2] * 100:.1f}%")
    print(f"  per step: {[round(v * 100, 1) for v in und[:-1]]}")

    bvs = summary["belief_vs_state"]
    if bvs:
        verb = "acted on" if cli.free_support else "suppressed"
        print("\n[mind-change-slm] === BELIEF vs already-committed STATE ===")
        print(f"  committed positions where the model wants another base: "
              f"{bvs['frac_committed_positions_where_belief_differs'] * 100:.2f}%")
        print(f"  positions the model ever wanted to revise : "
              f"{bvs['frac_positions_model_ever_wanted_to_revise'] * 100:.2f}%")
        print(f"  mean such revisions / sequence ({verb}) : "
              f"{bvs['mean_suppressed_revisions_per_sequence']:.1f}")

    rq = summary.get("revision_quality") or {}
    if rq:
        print("\n[mind-change-slm] === WERE THE REVISIONS GOOD? (FBCNN log p(class | seq)) ===")
        print(f"  events scored: {rq['n_events_scored']} over {rq['n_sequences']} sequences "
              f"(skipped, revision later undone: {rq['n_events_skipped_revision_undone']})")
        print(f"  CI95 = {rq['bootstrap']} -- events cluster by sequence, so a per-event "
              "binomial error bar is too narrow.")
        print(f"  {'':36}{'% improved (CI95)':>26}  {'mean delta (CI95)':>28}")
        for key, name in (
            ("revision", "revision vs its own old base"),
            ("null_random_token_same_position", "null: random base, same position"),
            ("null_random_position", "null: random base, random position"),
        ):
            b = rq[key]
            fl, fh = b["frac_improved_ci95"]
            ml, mh = b["mean_delta_ci95"]
            print(f"  {name:<36}{b['frac_improved'] * 100:7.2f} [{fl * 100:5.2f}, {fh * 100:5.2f}]"
                  f"  {b['mean_delta']:+11.4f} [{ml:+.4f}, {mh:+.4f}]")
        pr = rq["revision_vs_random_token_paired"]
        pl, ph = pr["frac_revision_better_ci95"]
        print(f"  paired (same event): revision beats a random alternative "
              f"{pr['frac_revision_better'] * 100:.2f}% [{pl * 100:.2f}, {ph * 100:.2f}]")
        lo, hi = rq["revision"]["frac_improved_ci95"]
        if lo <= 0.5 <= hi:
            print("  -> revisions are indistinguishable from a coin flip: the corrector is not correcting.")
        elif hi < 0.5:
            print("  -> revisions score *worse* than the base they replaced.")
        elif pr["mean_delta_difference_ci95"][0] <= 0.0:
            print("  -> revisions beat their old base, but no better than an arbitrary alternative "
                  "at the same position: the model picks *where*, not *what*.")
        else:
            print("  -> revisions beat both their old base and an arbitrary alternative: real corrections.")
        npos = rq["null_random_position"]
        if abs(npos["mean_delta"]) < 0.1 * npos["std_delta"]:
            print(
                f"  caveat: a random single-base substitution moves the score by only "
                f"{npos['mean_delta']:+.4f} nats (sd {npos['std_delta']:.4f}) -- this classifier "
                "barely responds to one-position edits, so the directions above can be resolvable "
                "yet negligible in magnitude."
            )
    elif cli.score_revisions:
        print("\n[mind-change-slm] === WERE THE REVISIONS GOOD? ===\n"
              "  nothing to score: this run produced 0 state-level revisions.")

    if summary["state"]["mean_switches_per_position"] == 0.0:
        if trainer != "routed_discrete" or corruption_mode != "independent":
            print("\n[mind-change-slm] note: 0 switches by construction -- this reverse step always "
                  "intersects with the current support, so a collapsed base is frozen. Control run.")
        elif not cli.free_support:
            print("\n[mind-change-slm] note: 0 switches. The support intersection is dropped after "
                  "60% of the steps, but the *logits* are still masked to the current support; "
                  "re-run with --free_support to let a collapsed position be re-predicted freely.")

    print(f"\n[mind-change-slm] wrote {json_path}")


if __name__ == "__main__":
    main()
