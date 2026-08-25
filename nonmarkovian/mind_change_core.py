"""Model-agnostic "did it change its mind?" metrics over an inference trajectory.

Shared by the MDLM evaluation (``nonmarkovian.mind_change_mdlm``) and the SLM/ShortListing
evaluation (``SLM/mind_change_slm.py``) so both report numbers that mean the same thing.

The only interface is a **state trajectory** ``[B, F, L]`` of uint8 token ids in which
``UNDECIDED`` (4) marks a position the sampler has not committed to yet. Each process supplies
its own notion of "committed":

* MDLM / absorbing state -- the position is unmasked (not ``[M]``).
* SLM / ShortListing     -- the position's shortlist has collapsed to a single token.

A *switch* (mind change) is a change between consecutive **committed** values at a position, with
undecided frames bridged: ``C C [?] C G`` is one switch (C -> G), no matter how many undecided
frames sit in between.

This module imports nothing but ``torch`` so it can be imported from either repository.
"""

from __future__ import annotations

import torch

NUC = ("A", "C", "G", "T")
_NO_COMMIT = 255  # sentinel: this position has never been resolved yet
UNDECIDED = 4     # "no committed token here": [M] for MDLM, |shortlist| > 1 for SLM


class MindChangeStats:
    """Streaming accumulator over batches of trajectories ``[B, F, L]`` (uint8, mask = UNDECIDED)."""

    def __init__(self, seq_len: int, num_frames: int, num_classes: int = 4) -> None:
        self.L = int(seq_len)
        self.F = int(num_frames)
        self.C = int(num_classes)
        self.n_seqs = 0
        # per-position sums over sequences
        self.pos_switches = torch.zeros(self.L, dtype=torch.float64)
        self.pos_switches_adj = torch.zeros(self.L, dtype=torch.float64)
        self.pos_remasks = torch.zeros(self.L, dtype=torch.float64)
        self.pos_ever_switched = torch.zeros(self.L, dtype=torch.float64)
        self.pos_net_changed = torch.zeros(self.L, dtype=torch.float64)
        # per-step (frame) sums over sequences x positions
        self.step_switches = torch.zeros(self.F, dtype=torch.float64)
        self.step_remasks = torch.zeros(self.F, dtype=torch.float64)
        self.step_unmasks = torch.zeros(self.F, dtype=torch.float64)
        self.step_mismatch = torch.zeros(self.F, dtype=torch.float64)  # frame != final sequence
        self.has_mismatch = False
        # distributions
        self.switch_matrix = torch.zeros(self.C * self.C, dtype=torch.float64)
        # re-mask -> refill events: did the position come back as the token it had before?
        self.refill_events = 0.0
        self.refill_restored = 0.0
        self.hist_switches = torch.zeros(1, dtype=torch.float64)   # grown on demand
        self.hist_distinct = torch.zeros(self.C + 1, dtype=torch.float64)
        # per-sequence rows for the CSV
        self.rows: list[tuple[int, float, float, float, float, float]] = []

    def _grow_hist(self, n: int) -> None:
        if n >= self.hist_switches.numel():
            pad = torch.zeros(n + 1 - self.hist_switches.numel(), dtype=torch.float64)
            self.hist_switches = torch.cat([self.hist_switches, pad])

    @torch.no_grad()
    def update(self, traj: torch.Tensor, final: torch.Tensor | None = None) -> None:
        """``traj``: ``[B, F, L]`` uint8 token ids (``UNDECIDED`` for masked). CPU.

        ``final`` ``[B, L]``: the sequence the run ended on. When given, each frame is also
        compared against it (``step_mismatch``) -- for the belief trajectory this is "how often
        the model's current guess disagrees with the answer it eventually lands on".
        """
        traj = traj.to("cpu")
        B, F_, L = traj.shape
        assert L == self.L and F_ == self.F, f"expected [*, {self.F}, {self.L}], got {tuple(traj.shape)}"

        last = torch.full((B, L), _NO_COMMIT, dtype=torch.uint8)     # last committed token
        first = torch.full((B, L), _NO_COMMIT, dtype=torch.uint8)    # first committed token
        seen = torch.zeros((B, L, self.C), dtype=torch.bool)         # tokens ever committed
        switches = torch.zeros((B, L), dtype=torch.int32)
        switches_adj = torch.zeros((B, L), dtype=torch.int32)
        remasks = torch.zeros((B, L), dtype=torch.int32)
        prev = None  # previous frame, raw (mask included)
        if final is not None:
            final = final.to("cpu", torch.uint8)
            self.has_mismatch = True

        for f in range(F_):
            cur = traj[:, f]
            committed = cur != UNDECIDED
            if final is not None:
                self.step_mismatch[f] += float((committed & (cur != final)).sum())

            # mind change: committed now, committed before, and different from the last commitment
            changed = committed & (last != _NO_COMMIT) & (cur != last)
            switches += changed.to(torch.int32)
            self.step_switches[f] += float(changed.sum())
            if changed.any():
                frm = last[changed].to(torch.long)
                to = cur[changed].to(torch.long)
                self.switch_matrix += torch.bincount(
                    frm * self.C + to, minlength=self.C * self.C
                ).to(torch.float64)

            if prev is not None:
                prev_committed = prev != UNDECIDED
                # a position coming back from [M] having been committed at some earlier frame
                refill = ~prev_committed & committed & (last != _NO_COMMIT)
                self.refill_events += float(refill.sum())
                self.refill_restored += float((refill & (cur == last)).sum())
                switches_adj += (changed & prev_committed).to(torch.int32)
                self.step_remasks[f] += float((prev_committed & ~committed).sum())
                self.step_unmasks[f] += float((~prev_committed & committed).sum())
                remasks += (prev_committed & ~committed).to(torch.int32)

            first = torch.where(committed & (first == _NO_COMMIT), cur, first)
            idx = torch.where(committed, cur.to(torch.long), torch.zeros_like(cur, dtype=torch.long))
            seen |= torch.nn.functional.one_hot(idx, self.C).bool() & committed.unsqueeze(-1)
            last = torch.where(committed, cur, last)
            prev = cur

        final_seq = final if final is not None else traj[:, -1]
        net_changed = (first != _NO_COMMIT) & (final_seq != UNDECIDED) & (final_seq != first)
        distinct = seen.sum(dim=-1)  # [B, L], 0..C

        self.n_seqs += int(B)
        self.pos_switches += switches.sum(dim=0).to(torch.float64)
        self.pos_switches_adj += switches_adj.sum(dim=0).to(torch.float64)
        self.pos_remasks += remasks.sum(dim=0).to(torch.float64)
        self.pos_ever_switched += (switches > 0).sum(dim=0).to(torch.float64)
        self.pos_net_changed += net_changed.sum(dim=0).to(torch.float64)

        self._grow_hist(int(switches.max()))
        self.hist_switches += torch.bincount(
            switches.reshape(-1).to(torch.long), minlength=self.hist_switches.numel()
        ).to(torch.float64)
        self.hist_distinct += torch.bincount(
            distinct.reshape(-1).to(torch.long), minlength=self.C + 1
        ).to(torch.float64)

        base = self.n_seqs - int(B)
        sw_seq = switches.sum(dim=1).to(torch.float64)
        for i in range(B):
            self.rows.append(
                (
                    base + i,
                    float(sw_seq[i]),
                    float((switches[i] > 0).sum()),
                    float(switches_adj[i].sum()),
                    float(remasks[i].sum()),
                    float(net_changed[i].sum()),
                )
            )

    def summary(self) -> dict:
        n = max(self.n_seqs, 1)
        n_pos = float(n * self.L)
        hist = self.hist_switches
        counts = torch.arange(hist.numel(), dtype=torch.float64)
        mat = self.switch_matrix.reshape(self.C, self.C)
        out = {
            "n_sequences": self.n_seqs,
            "seq_len": self.L,
            "num_frames": self.F,
            # --- headline numbers ---
            "mean_switches_per_position": float(self.pos_switches.sum() / n_pos),
            "mean_switches_per_sequence": float(self.pos_switches.sum() / n),
            "frac_positions_ever_changed": float(self.pos_ever_switched.sum() / n_pos),
            "frac_positions_net_changed": float(self.pos_net_changed.sum() / n_pos),
            "mean_switches_adjacent_per_position": float(self.pos_switches_adj.sum() / n_pos),
            "mean_remasks_per_position": float(self.pos_remasks.sum() / n_pos),
            "remask_refill_events": self.refill_events,
            "frac_refills_restoring_same_token": (
                self.refill_restored / self.refill_events if self.refill_events else 0.0
            ),
            "max_switches_at_a_position": int((hist > 0).nonzero().max()) if float(hist.sum()) else 0,
            "mean_distinct_tokens_per_position": float(
                (self.hist_distinct * torch.arange(self.C + 1, dtype=torch.float64)).sum()
                / max(float(self.hist_distinct.sum()), 1.0)
            ),
            # --- distributions ---
            "hist_switches_per_position": {
                str(int(k)): float(v) for k, v in zip(counts.tolist(), hist.tolist()) if v > 0
            },
            "frac_positions_by_switch_count": {
                str(int(k)): float(v) / n_pos for k, v in zip(counts.tolist(), hist.tolist()) if v > 0
            },
            "frac_positions_by_distinct_tokens": {
                str(i): float(self.hist_distinct[i]) / n_pos for i in range(self.C + 1)
            },
            "switch_matrix_counts": {
                NUC[i]: {NUC[j]: float(mat[i, j]) for j in range(self.C)} for i in range(self.C)
            },
            # --- per-step curves (mean events per position, per step) ---
            "switches_per_step": (self.step_switches / n_pos).tolist(),
            "remasks_per_step": (self.step_remasks / n_pos).tolist(),
            "unmasks_per_step": (self.step_unmasks / n_pos).tolist(),
        }
        if self.has_mismatch:
            out["mismatch_with_final_per_step"] = (self.step_mismatch / n_pos).tolist()
            out["frac_positions_belief_final_mismatch"] = float(
                self.step_mismatch.sum() / (n_pos * self.F)
            )
        return out


@torch.no_grad()
def _revision_events(traj: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Every mind-change event in a state trajectory ``[B, F, L]``.

    Returns 1-D tensors ``(seq, pos, old, new)``: at frame-time the position ``pos`` of sequence
    ``seq`` went from its previously committed token ``old`` to ``new``. Mask frames are bridged
    (``old`` is the last *committed* token), so re-mask -> refill-with-a-different-token counts.
    """
    B, F_, L = traj.shape
    last = torch.full((B, L), _NO_COMMIT, dtype=torch.uint8)
    seqs, poss, olds, news = [], [], [], []
    for f in range(F_):
        cur = traj[:, f]
        committed = cur != UNDECIDED
        changed = committed & (last != _NO_COMMIT) & (cur != last)
        if changed.any():
            idx = changed.nonzero(as_tuple=False)
            seqs.append(idx[:, 0])
            poss.append(idx[:, 1])
            olds.append(last[changed].long())
            news.append(cur[changed].long())
        last = torch.where(committed, cur, last)
    if not seqs:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty, empty, empty
    return (
        torch.cat(seqs).long(),
        torch.cat(poss).long(),
        torch.cat(olds),
        torch.cat(news),
    )


def _pick_excluding(excluded: torch.Tensor, gen: torch.Generator, num_classes: int = 4) -> torch.Tensor:
    """Uniform token in ``0..C-1`` avoiding the columns marked True in ``excluded`` ``[N, C]``."""
    n = excluded.shape[0]
    allowed = ~excluded
    k = allowed.sum(dim=1)                                  # how many choices per row
    rnd = torch.rand(n, generator=gen)
    pick = torch.minimum((rnd * k.to(torch.float32)).floor().long(), k - 1) + 1
    rank = allowed.long().cumsum(dim=1)                     # 1..k at the allowed columns
    hit = allowed & (rank == pick.unsqueeze(1))
    return hit.float().argmax(dim=1)


class RevisionScorer:
    """Was each revision a *good* change? Scores counterfactuals under the FBCNN classifier.

    For every revision event ``pos: old -> ...`` the model's final sequence is compared against
    the same sequence with ``pos`` set back to ``old``, using ``log p(target class | sequence)``:

    ``delta_revision = s(final) - s(final with the pre-revision token restored)``

    ``> 0`` means the revision raised the conditional class probability. Two nulls are scored on
    exactly the same events, because a bare "% improved" above 50% means nothing on its own:

    ``delta_random_token``
        same position, but a random token that is neither ``old`` nor the final one -- did the
        model know *what* to write, given it chose *where*?
    ``delta_random_position``
        a random position, random different token -- the typical cost of any single substitution,
        which calibrates the scale of the other two.
    """

    def __init__(self, score_fn, seed: int = 0, num_classes: int = 4, chunk: int = 1024) -> None:
        """``score_fn(seqs [N, L] ids, labels [N]) -> [N]`` log-probabilities, CPU tensors in and out.

        Each repository supplies its own ``score_fn`` (both wrap the same FBCNN classifier, but
        load it differently), which keeps this module free of model imports.
        """
        self.score_fn = score_fn
        self.C = int(num_classes)
        self.chunk = int(chunk)
        self.gen = torch.Generator().manual_seed(int(seed) + 20260812)
        self.rows: list[tuple] = []
        self.n_events = 0
        self.n_skipped = 0

    @torch.no_grad()
    def _score(self, seqs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """``log p(label | seq)`` for ``seqs`` ``[N, L]`` ids and ``labels`` ``[N]``."""
        out = []
        for i in range(0, seqs.shape[0], self.chunk):
            out.append(
                self.score_fn(
                    seqs[i : i + self.chunk].long().clamp(0, self.C - 1),
                    labels[i : i + self.chunk].long(),
                ).detach().cpu().float()
            )
        return torch.cat(out) if out else torch.zeros(0)

    @torch.no_grad()
    def update(
        self,
        traj: torch.Tensor,
        x_final: torch.Tensor,
        labels: torch.Tensor,
        *,
        max_events: int = 0,
        seq_offset: int = 0,
    ) -> None:
        seq, pos, old, new = _revision_events(traj.to("cpu"))
        if seq.numel() == 0:
            return
        final = x_final.detach().to("cpu").long().clamp(0, self.C - 1)   # [B, L]
        lab = labels.detach().to("cpu").long()

        # A revision that was later undone leaves final[pos] == old: the counterfactual would be
        # identical to the actual sequence, so the comparison carries no information.
        cur_tok = final[seq, pos]
        keep = cur_tok != old
        self.n_skipped += int((~keep).sum())
        seq, pos, old, new, cur_tok = seq[keep], pos[keep], old[keep], new[keep], cur_tok[keep]
        if seq.numel() == 0:
            return
        if max_events > 0 and seq.numel() > max_events:
            sel = torch.randperm(seq.numel(), generator=self.gen)[:max_events]
            seq, pos, old, new, cur_tok = seq[sel], pos[sel], old[sel], new[sel], cur_tok[sel]

        n, L = seq.numel(), final.shape[1]
        ev_lab = lab[seq]
        ar = torch.arange(n)

        # actual final sequence (scored once per event, cheap enough and keeps the code simple)
        base = self._score(final[seq], ev_lab)

        # (1) revert the revision
        cf = final[seq].clone()
        cf[ar, pos] = old
        s_rev = self._score(cf, ev_lab)

        # (2) same position, a random token that is neither `old` nor the final token
        excl = torch.zeros(n, self.C, dtype=torch.bool)
        excl[ar, old] = True
        excl[ar, cur_tok] = True
        rand_tok = _pick_excluding(excl, self.gen, self.C)
        cf = final[seq].clone()
        cf[ar, pos] = rand_tok
        s_tok = self._score(cf, ev_lab)

        # (3) a random position, a random different token
        rpos = (torch.rand(n, generator=self.gen) * L).floor().long().clamp(max=L - 1)
        rcur = final[seq, rpos]
        excl = torch.zeros(n, self.C, dtype=torch.bool)
        excl[ar, rcur] = True
        rand_tok2 = _pick_excluding(excl, self.gen, self.C)
        cf = final[seq].clone()
        cf[ar, rpos] = rand_tok2
        s_pos = self._score(cf, ev_lab)

        d_rev, d_tok, d_pos = base - s_rev, base - s_tok, base - s_pos
        self.n_events += n
        for i in range(n):
            self.rows.append(
                (
                    seq_offset + int(seq[i]),
                    int(pos[i]),
                    NUC[int(old[i])],
                    NUC[int(new[i])],
                    NUC[int(cur_tok[i])],
                    float(d_rev[i]),
                    float(d_tok[i]),
                    float(d_pos[i]),
                )
            )

    def summary(self, n_boot: int = 1000) -> dict:
        if not self.rows:
            return {}
        d_rev = torch.tensor([r[5] for r in self.rows], dtype=torch.float64)
        d_tok = torch.tensor([r[6] for r in self.rows], dtype=torch.float64)
        d_pos = torch.tensor([r[7] for r in self.rows], dtype=torch.float64)
        seq_ids = torch.tensor([r[0] for r in self.rows], dtype=torch.long)

        # Events are clustered: ~50 per sequence, all sharing one context and one label, so a
        # binomial se over events assumes an independence that does not hold (measured ~1.3x too
        # narrow at 32 sequences, and the gap grows with events-per-sequence). Resample whole
        # *sequences* instead -- the number of sequences, not events, is what buys precision.
        _uniq, inv = torch.unique(seq_ids, return_inverse=True)
        groups = [(inv == i).nonzero(as_tuple=True)[0] for i in range(int(inv.max()) + 1)]
        n_clusters = len(groups)
        boot_gen = torch.Generator().manual_seed(20260812)
        boot_idx = [
            torch.cat([groups[int(p)] for p in torch.randint(n_clusters, (n_clusters,), generator=boot_gen)])
            for _ in range(int(n_boot))
        ]

        def ci95(vals: list[float]) -> list[float]:
            v = torch.tensor(vals, dtype=torch.float64).sort().values
            lo = v[max(int(0.025 * len(vals)) - 1, 0)]
            hi = v[min(int(0.975 * len(vals)), len(vals) - 1)]
            return [float(lo), float(hi)]

        def block(d: torch.Tensor) -> dict:
            n = d.numel()
            frac = float((d > 0).sum()) / n
            fr_b = [float((d[i] > 0).sum()) / d[i].numel() for i in boot_idx]
            mn_b = [float(d[i].mean()) for i in boot_idx]
            return {
                "n": n,
                "n_sequences": n_clusters,
                "frac_improved": frac,
                "frac_improved_ci95": ci95(fr_b),
                "frac_improved_stderr_clustered": float(torch.tensor(fr_b).std()),
                "frac_improved_stderr_naive_binomial": float((frac * (1 - frac) / n) ** 0.5),
                "mean_delta": float(d.mean()),
                "mean_delta_ci95": ci95(mn_b),
                "median_delta": float(d.median()),
                "std_delta": float(d.std(unbiased=True)),
            }

        paired = d_rev - d_tok
        pf_b = [float((paired[i] > 0).sum()) / paired[i].numel() for i in boot_idx]
        pm_b = [float(paired[i].mean()) for i in boot_idx]
        return {
            "n_events_scored": self.n_events,
            "n_events_skipped_revision_undone": self.n_skipped,
            "n_sequences": n_clusters,
            "bootstrap": f"{int(n_boot)} resamples over whole sequences (clustered)",
            "revision": block(d_rev),
            "null_random_token_same_position": block(d_tok),
            "null_random_position": block(d_pos),
            "revision_vs_random_token_paired": {
                "frac_revision_better": float((paired > 0).sum()) / paired.numel(),
                "frac_revision_better_ci95": ci95(pf_b),
                "mean_delta_difference": float(paired.mean()),
                "mean_delta_difference_ci95": ci95(pm_b),
            },
        }


class BeliefVsStateStats:
    """How often the model's belief contradicts a token the sampler has *already* committed.

    ``preds[:, i]`` is the belief computed from state frame ``states[:, i]``, so the two line up
    directly. Only positions that are already unmasked at that frame are counted -- a disagreement
    there is a mind change the support mask suppresses (or, with ``--free_support``, acts on).
    """

    def __init__(self, seq_len: int) -> None:
        self.L = int(seq_len)
        self.n_seqs = 0
        self.step_disagree: torch.Tensor | None = None
        self.step_committed: torch.Tensor | None = None
        self.pos_ever_disagreed = torch.zeros(self.L, dtype=torch.float64)
        self.pos_disagree = torch.zeros(self.L, dtype=torch.float64)
        self.pos_committed = torch.zeros(self.L, dtype=torch.float64)

    @torch.no_grad()
    def update(self, states: torch.Tensor, preds: torch.Tensor) -> None:
        states, preds = states.to("cpu"), preds.to("cpu")
        B, T, L = preds.shape
        if self.step_disagree is None:
            self.step_disagree = torch.zeros(T, dtype=torch.float64)
            self.step_committed = torch.zeros(T, dtype=torch.float64)
        st = states[:, :T]                       # frame i is the input to step i
        committed = st != UNDECIDED
        disagree = committed & (preds != st)
        self.step_disagree += disagree.sum(dim=(0, 2)).to(torch.float64)
        self.step_committed += committed.sum(dim=(0, 2)).to(torch.float64)
        self.pos_disagree += disagree.sum(dim=(0, 1)).to(torch.float64)
        self.pos_committed += committed.sum(dim=(0, 1)).to(torch.float64)
        self.pos_ever_disagreed += disagree.any(dim=1).sum(dim=0).to(torch.float64)
        self.n_seqs += int(B)

    def summary(self) -> dict:
        if self.step_disagree is None:
            return {}
        n_pos = float(max(self.n_seqs, 1) * self.L)
        tot_com = float(self.pos_committed.sum())
        return {
            "frac_committed_positions_where_belief_differs": float(self.pos_disagree.sum())
            / max(tot_com, 1.0),
            "frac_positions_model_ever_wanted_to_revise": float(self.pos_ever_disagreed.sum()) / n_pos,
            "mean_suppressed_revisions_per_sequence": float(self.pos_disagree.sum())
            / max(self.n_seqs, 1),
            "disagree_per_step": (
                self.step_disagree / self.step_committed.clamp(min=1.0)
            ).tolist(),
            "committed_fraction_per_step": (self.step_committed / n_pos).tolist(),
        }




class EmbeddingProximity:
    """How close is a sequence to the *real* data in the FBD classifier's embedding space?

    FBD's own logic applied per sequence. The class head of these classifiers is unreliable (on
    fly brain, real held-out sequences score below chance under their own labels at ~10% top-1),
    but the embedding is exactly what FBD uses and it does separate real from random. So instead of
    ``log p(class | seq)`` we score

        ``-(e - mu)^T Sigma^-1 (e - mu)``

    the negated squared Mahalanobis distance to the real-embedding distribution, so **higher is
    closer to the real manifold** and deltas keep the same "positive = better" reading as before.

    ``mu`` is either the global real mean (``class_conditional=False``: "does it look like real
    DNA?") or the mean of the real sequences carrying the same label (``True``: "does it look like
    real DNA *of this class*?"), falling back to the global mean for classes with fewer than
    ``min_per_class`` references. ``Sigma`` is always pooled over all references, since per-class
    covariances are not estimable at 81 classes.
    """

    def __init__(
        self,
        embed_fn,
        ref_embeddings: torch.Tensor,
        ref_labels: torch.Tensor | None = None,
        *,
        class_conditional: bool = False,
        ridge: float = 1e-3,
        min_per_class: int = 20,
    ) -> None:
        self.embed_fn = embed_fn
        self.class_conditional = bool(class_conditional)
        E = ref_embeddings.detach().to(torch.float64)
        self.n_ref, d = E.shape
        self.mu = E.mean(dim=0)
        cov = torch.cov(E.T)
        # ridge relative to the average variance keeps the inverse well conditioned at d=128
        cov = cov + (ridge * torch.diagonal(cov).mean()) * torch.eye(d, dtype=cov.dtype)
        self.prec = torch.linalg.pinv(cov)
        self.class_mu: dict[int, torch.Tensor] = {}
        if class_conditional and ref_labels is not None:
            lab = ref_labels.detach().reshape(-1).long()
            for c in lab.unique().tolist():
                sel = lab == c
                if int(sel.sum()) >= int(min_per_class):
                    self.class_mu[int(c)] = E[sel].mean(dim=0)
        self.n_classes_with_mean = len(self.class_mu)

    @torch.no_grad()
    def __call__(self, seqs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        e = self.embed_fn(seqs).detach().to(torch.float64)
        if self.class_mu:
            mu = torch.stack(
                [self.class_mu.get(int(c), self.mu) for c in labels.reshape(-1).tolist()]
            )
        else:
            mu = self.mu.unsqueeze(0)
        diff = e - mu
        return -((diff @ self.prec) * diff).sum(dim=-1)
