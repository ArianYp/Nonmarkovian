"""How often does the model *change its mind* during MDLM inference?

Experiment: run the reverse (inference) process, record the full trajectory of token states, and
count how many times each position revises an already-committed nucleotide -- e.g. a position that
reads ``C`` (one-hot ``0,1,0,0``) at some step and ``G`` (``0,0,1,0``) at a later step.

A "mind change" (**switch**) is counted on the *committed* sub-sequence of a position, i.e. the
trajectory with the ``[M]`` frames removed::

    trajectory:   [M] [M]  C   C  [M]  C   G   G      committed: C C C G  -> 1 switch (C -> G)

So changes need **not** happen in consecutive steps: a position may commit to ``C``, get re-masked,
and later come back as ``G`` -- that still counts. The stricter "adjacent" variant (both frames
non-mask, no intervening mask) is reported separately as a subset.

Two trajectories are analysed with the same metric:

``state``
    the *sampled* sequence ``x_t`` at each reverse step -- what the model actually commits to.
``belief``
    ``argmax p_theta(x_0 | x_t)`` at each step, read off **before** the support mask -- what the
    model would say the clean sequence is, whether or not it is allowed to act on it.

Both matter, because of how ``sample_mdlm`` samples: the logits are restricted to the support of
the current state, so a position that is already unmasked can only be re-sampled *to itself*. A
committed token can therefore never be revised in place -- the only route to a state-level switch
is re-mask -> refill with a different token. Pass ``--free_support`` to lift that restriction and
let the non-Markovian corrector phase overwrite committed tokens directly; the ``belief`` metrics
are unaffected by the flag and measure the model's own mind changes either way.

For state switches the sampler matters: with strict carry-over (``--corruption_mode trajectory``,
or the ``simple_mdlm`` baseline) a position is frozen once unmasked, so every state switch count is
0 by construction -- that run is the control.

Metrics reported (per position, aggregated over sequences):

* ``switches``            -- number of mind changes (mask-bridging definition, primary)
* ``switches_adjacent``   -- subset where the change is between two consecutive non-mask frames
* ``distinct_tokens``     -- how many of A/C/G/T a position ever committed to (1..4)
* ``net_changed``         -- first committed token != final token (a *net* mind change)
* ``remasks``             -- committed token -> ``[M]`` events (the mechanism enabling revision)
* per-step curves of switches / remasks / unmaskings (shows the corrector phase kicking in)
* a 4x4 ``from -> to`` switch matrix (e.g. how often C -> G specifically)

plus, for the belief trajectory, ``frac_positions_belief_final_mismatch`` -- how often the model's
belief at a step disagrees with the token that ends up in the final sequence -- and a
``belief_vs_state`` block: how often the belief contradicts a token the sampler has *already*
committed, i.e. a revision the support mask suppresses.

Caveat on ``belief_vs_state``: the MDLM NELBO only supervises **masked** positions, so a model's
logits at already-unmasked positions may simply be untrained. A disagreement rate near the 75%
chance level means "unsupervised output", not "the model wants to revise"; a rate near 0 means the
model has learned to copy what is already there. Compare models before reading it as a wish to
revise.

``--score_revisions`` additionally answers *were the changes good?*. For every revision event the
model's final sequence is scored against counterfactuals under the FBCNN classifier, using
``log p(target class | sequence)``:

* ``revision``              -- final vs the same sequence with the pre-revision token restored.
  ``> 50%`` improved means the new token beats the one it replaced.
* ``null: random token, same position`` -- final vs a random token (neither the old nor the final
  one) at the *same* position: did the model know **what** to write, given it chose **where**?
* ``null: random position``  -- final vs a random token at a random position: the typical cost of
  any single substitution, which calibrates the scale of the other two.

A bare "% improved" above 50 means nothing without those nulls, which is why all three are scored
on the same events.

Usage::

    python -m nonmarkovian.mind_change_mdlm \
        --checkpoint checkpoints/routed_mdlm.best_fbd.pt \
        --n_samples 256 --out_dir logs/mind_change/routed

    # ... and judge whether the revisions improved the sequences
    python -m nonmarkovian.mind_change_mdlm \
        --checkpoint checkpoints/routed_mdlm_mel.best.pt --dfm_melanoma \
        --n_samples 256 --num_timesteps_sample 32 --guidance_scale 0.5 \
        --score_revisions --fbcnn_ckpt fbd_mel.ckpt --fbcnn_num_cls 47 \
        --out_dir logs/mind_change --tag nm

    # Markovian control (expect all-zero switch counts)
    python -m nonmarkovian.mind_change_mdlm \
        --checkpoint checkpoints/routed_mdlm.best_fbd.pt \
        --corruption_mode trajectory --n_samples 256 --out_dir logs/mind_change/markovian

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
)
from nonmarkovian.eval_checkpoint_mdlm import _detect_trainer_mdlm
from nonmarkovian.sample_mdlm import sample_sequences_mdlm
from nonmarkovian.sample_simple_mdlm import sample_sequences_simple_mdlm
from nonmarkovian.validation import _use_conditional_sampling_labels
from nonmarkovian.mind_change_core import (
    NUC,
    BeliefVsStateStats,
    MindChangeStats,
    RevisionScorer,
)
from nonmarkovian.vocab import MASK_IDX

def main() -> None:
    p = argparse.ArgumentParser(
        description="Count how often the model revises a committed token during MDLM inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--trainer", type=str, default="", choices=("", "routed_mdlm", "simple_mdlm"))
    p.add_argument("--n_samples", type=int, default=256, help="Number of sequences to generate.")
    p.add_argument("--batch", type=int, default=64, help="Sampling batch size.")
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
        help="Routed-only reverse-step constraint. 'trajectory' = Markovian control (0 switches).",
    )
    p.add_argument("--independent_threshold", type=float, default=-1.0, help="<0 = ckpt value.")
    p.add_argument("--guidance_scale", type=float, default=0.0)
    p.add_argument(
        "--label", type=int, default=-1,
        help="Force one conditioning class for every sample; -1 = take labels from the data split.",
    )
    p.add_argument(
        "--unconditional", action="store_true",
        help="Sample without labels even if the checkpoint is conditional.",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="logs/mind_change")
    p.add_argument("--tag", type=str, default="", help="Prefix for the output filenames.")
    p.add_argument(
        "--save_trajectories", type=int, default=0,
        help="Also dump the raw token trajectories of the first N sequences (.pt).",
    )
    p.add_argument(
        "--free_support", action="store_true",
        help="Routed-only: drop the support mask so the corrector phase may overwrite an "
        "already-committed token in place (default keeps the shipped behaviour, where a "
        "committed token can only change via a re-mask).",
    )
    p.add_argument(
        "--score_revisions", action="store_true",
        help="Score every revision against counterfactuals with the FBCNN classifier: was the "
        "new token better than the one it replaced? Requires --fbcnn_ckpt and a conditional "
        "checkpoint (the score is log p(target class | sequence)).",
    )
    p.add_argument("--fbcnn_ckpt", type=str, default="", help="FBCNN classifier for --score_revisions.")
    p.add_argument("--fbcnn_num_cls", type=int, default=0)
    p.add_argument("--fbcnn_stacks", type=int, default=0)
    p.add_argument(
        "--score_max_events", type=int, default=0,
        help="Cap on revision events scored per batch (0 = all).",
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

    trainer = cli.trainer.strip().lower() or _detect_trainer_mdlm(ckpt)
    if cli.free_support and trainer != "routed_mdlm":
        print("[mind-change] warning: --free_support only applies to routed_mdlm; ignoring.")

    overrides: dict = {"guidance_scale": float(cli.guidance_scale), "seed": int(cli.seed)}
    if cli.num_timesteps_sample > 0:
        overrides["num_timesteps_sample"] = int(cli.num_timesteps_sample)
    if cli.history_mode:
        overrides["history_mode"] = cli.history_mode
    if cli.corruption_mode:
        overrides["corruption_mode"] = cli.corruption_mode
    if cli.independent_threshold >= 0.0:
        overrides["independent_threshold"] = float(cli.independent_threshold)
    if cli.unconditional:
        overrides["no_labels"] = True
    args = _build_args_namespace(cfg, overrides)

    model = _build_routed_model(cfg, device) if trainer == "routed_mdlm" else _build_simple_model(cfg, device)
    if trainer == "routed_mdlm":
        model.num_timesteps = int(args.num_timesteps_sample)
    model.load_state_dict(state, strict=not cli.no_strict_load)
    model.eval()

    T = int(args.num_timesteps_sample)
    seq_len = int(cli.seq_len) if cli.seq_len > 0 else int(getattr(args, "max_len", 500))
    scheduler = str(getattr(args, "bernoulli_scheduler", "loglinear"))
    corruption_mode = str(getattr(args, "corruption_mode", "independent"))
    independent_threshold = float(getattr(args, "independent_threshold", 0.6))
    use_labs = _use_conditional_sampling_labels(args)

    scorer = None
    if cli.score_revisions:
        if not cli.fbcnn_ckpt.strip():
            raise SystemExit("--score_revisions needs --fbcnn_ckpt (e.g. fbd_mel.ckpt).")
        if not use_labs:
            raise SystemExit(
                "--score_revisions scores log p(target class | sequence), so it needs a "
                "conditional checkpoint sampled with labels (drop --unconditional)."
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

        scorer = RevisionScorer(_score_fn, seed=int(cli.seed))

    label_iter = None
    if use_labs and cli.label < 0:
        loader = _build_loader(
            cfg, cli.split, batch_size=int(cli.batch),
            dfm_root_override=cli.dfm_enhancer, melanoma_override=cli.dfm_melanoma,
        )
        label_iter = iter(loader)

    print(
        f"[mind-change] checkpoint={ckpt_path.resolve()} trainer={trainer}\n"
        f"[mind-change] n_samples={cli.n_samples} seq_len={seq_len} steps={T} "
        f"scheduler={scheduler} corruption_mode={corruption_mode if trainer == 'routed_mdlm' else 'carry-over'} "
        f"independent_threshold={independent_threshold} history_mode={getattr(args, 'history_mode', 'n/a')} "
        f"guidance_scale={float(args.guidance_scale)} labels={'yes' if use_labs else 'no'} "
        f"support_constraint={'off (--free_support)' if cli.free_support else 'on'}"
    )

    stats = MindChangeStats(seq_len=seq_len, num_frames=T + 2)
    stats_belief = MindChangeStats(seq_len=seq_len, num_frames=T)
    stats_bvs = BeliefVsStateStats(seq_len=seq_len)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cli.seed))
    kept_traj: list[torch.Tensor] = []
    kept_pred: list[torch.Tensor] = []

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
                    label_iter = iter(
                        _build_loader(
                            cfg, cli.split, batch_size=int(cli.batch),
                            dfm_root_override=cli.dfm_enhancer, melanoma_override=cli.dfm_melanoma,
                        )
                    )
                    batch = next(label_iter)
                lab = batch.get("label")
                if lab is not None:
                    lab = lab.to(device)
                    if lab.shape[0] < take:  # last, short batch
                        reps = (take + lab.shape[0] - 1) // lab.shape[0]
                        lab = lab.repeat(reps)
                    labels = lab[:take]

        if trainer == "routed_mdlm":
            x_final, traj, pred = sample_sequences_mdlm(
                model, T, take, seq_len, device,
                num_timesteps_train=int(getattr(args, "num_timesteps", T)),
                labels=labels,
                guidance_scale=float(args.guidance_scale),
                scheduler=scheduler,
                generator=gen,
                history_mode=str(getattr(args, "history_mode", "trajectory")),
                corruption_mode=corruption_mode,
                independent_threshold=independent_threshold,
                return_trajectory=True,
                support_constraint=not cli.free_support,
            )
        else:
            x_final, traj, pred = sample_sequences_simple_mdlm(
                model, T, take, seq_len, device,
                num_timesteps_train=int(getattr(args, "num_timesteps", T)),
                labels=labels,
                guidance_scale=float(args.guidance_scale),
                scheduler=scheduler,
                generator=gen,
                return_trajectory=True,
            )
        stats.update(traj)
        stats_belief.update(pred, final=x_final.to("cpu", torch.uint8))
        stats_bvs.update(traj, pred)
        if scorer is not None and labels is not None:
            scorer.update(
                traj, x_final, labels,
                max_events=int(cli.score_max_events), seq_offset=collected,
            )
        if cli.save_trajectories > 0 and sum(t.shape[0] for t in kept_traj) < cli.save_trajectories:
            need = cli.save_trajectories - sum(t.shape[0] for t in kept_traj)
            kept_traj.append(traj[:need].clone())
            kept_pred.append(pred[:need].clone())
        collected += take
        print(f"[mind-change] {collected}/{cli.n_samples} sequences", flush=True)

    summary = {
        "state": stats.summary(),
        "belief": stats_belief.summary(),
        "belief_vs_state": stats_bvs.summary(),
    }
    if scorer is not None:
        summary["revision_quality"] = scorer.summary()
    summary["config"] = {
        "checkpoint": str(ckpt_path.resolve()),
        "trainer": trainer,
        "num_timesteps_sample": T,
        "scheduler": scheduler,
        "corruption_mode": corruption_mode if trainer == "routed_mdlm" else "carry_over",
        "independent_threshold": independent_threshold,
        "history_mode": str(getattr(args, "history_mode", "n/a")),
        "guidance_scale": float(args.guidance_scale),
        "label": int(cli.label),
        "conditional": bool(use_labs),
        "seed": int(cli.seed),
        "split": cli.split,
        "support_constraint": not bool(cli.free_support),
    }

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (cli.tag + "_") if cli.tag else ""
    json_path = out_dir / f"{tag}mind_change.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_dir / f"{tag}mind_change_per_sequence.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "seq_idx,state_n_switches,state_n_positions_changed,state_n_switches_adjacent,"
            "state_n_remasks,state_n_net_changed,belief_n_switches,belief_n_positions_changed,"
            "belief_n_net_changed\n"
        )
        for r, rb in zip(stats.rows, stats_belief.rows):
            f.write(
                f"{r[0]},{r[1]:.0f},{r[2]:.0f},{r[3]:.0f},{r[4]:.0f},{r[5]:.0f},"
                f"{rb[1]:.0f},{rb[2]:.0f},{rb[5]:.0f}\n"
            )

    pos_path = out_dir / f"{tag}mind_change_per_position.csv"
    n = max(stats.n_seqs, 1)
    with open(pos_path, "w", encoding="utf-8") as f:
        f.write(
            "position,state_mean_switches,state_frac_ever_changed,state_mean_remasks,"
            "state_frac_net_changed,belief_mean_switches,belief_frac_ever_changed\n"
        )
        for j in range(stats.L):
            f.write(
                f"{j},{stats.pos_switches[j] / n:.6f},{stats.pos_ever_switched[j] / n:.6f},"
                f"{stats.pos_remasks[j] / n:.6f},{stats.pos_net_changed[j] / n:.6f},"
                f"{stats_belief.pos_switches[j] / n:.6f},{stats_belief.pos_ever_switched[j] / n:.6f}\n"
            )

    if scorer is not None and scorer.rows:
        rev_path = out_dir / f"{tag}revision_scores.csv"
        with open(rev_path, "w", encoding="utf-8") as f:
            f.write("seq_idx,position,from,to,final,delta_revision,delta_random_token,delta_random_position\n")
            for r in scorer.rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.6f},{r[6]:.6f},{r[7]:.6f}\n")

    if kept_traj:
        traj_path = out_dir / f"{tag}trajectories.pt"
        torch.save(
            {"states": torch.cat(kept_traj, dim=0), "beliefs": torch.cat(kept_pred, dim=0)},
            traj_path,
        )

    def _report(name: str, s: dict, *, remasks: bool) -> None:
        print(f"\n[mind-change] === {name} ===")
        print(f"  sequences                        : {s['n_sequences']}  (L={s['seq_len']}, frames={s['num_frames']})")
        print(f"  mean switches / position         : {s['mean_switches_per_position']:.4f}")
        print(f"  mean switches / sequence         : {s['mean_switches_per_sequence']:.2f}")
        print(f"  positions ever changed           : {s['frac_positions_ever_changed'] * 100:.2f}%")
        print(f"  positions net-changed (1st!=final): {s['frac_positions_net_changed'] * 100:.2f}%")
        print(f"  adjacent-frame flips             : {s['mean_switches_adjacent_per_position']:.4f} / position")
        if remasks:
            print(f"  mean re-maskings / position      : {s['mean_remasks_per_position']:.4f}")
            if s.get("remask_refill_events"):
                print(
                    f"  re-mask -> refill events         : {s['remask_refill_events']:.0f}, of which "
                    f"{s['frac_refills_restoring_same_token'] * 100:.1f}% came back as the same token"
                )
        if "frac_positions_belief_final_mismatch" in s:
            print(f"  belief != final token (all steps): {s['frac_positions_belief_final_mismatch'] * 100:.2f}%")
        print(f"  max switches at one position     : {s['max_switches_at_a_position']}")
        print(f"  mean distinct tokens visited     : {s['mean_distinct_tokens_per_position']:.4f}")
        print("  switch-count distribution over positions:")
        for k, v in sorted(s["frac_positions_by_switch_count"].items(), key=lambda kv: int(kv[0]))[:8]:
            print(f"    {k:>3} switches: {v * 100:6.2f}%")
        print("  from -> to switch matrix (counts):")
        print("        " + "  ".join(f"{c:>10}" for c in NUC))
        for c in NUC:
            row = s["switch_matrix_counts"][c]
            print(f"    {c} " + "  ".join(f"{row[d]:>10.0f}" for d in NUC))

    _report("STATE trajectory (what the sampler committed to)", summary["state"], remasks=True)
    _report("BELIEF trajectory (argmax p(x0|xt), unconstrained)", summary["belief"], remasks=False)

    bvs = summary["belief_vs_state"]
    if bvs:
        verb = "acted on" if cli.free_support else "suppressed"
        print("\n[mind-change] === BELIEF vs already-committed STATE ===")
        print(
            f"  committed positions where the model wants another token: "
            f"{bvs['frac_committed_positions_where_belief_differs'] * 100:.2f}%"
        )
        print(
            f"  positions the model ever wanted to revise : "
            f"{bvs['frac_positions_model_ever_wanted_to_revise'] * 100:.2f}%"
        )
        print(f"  mean such revisions / sequence ({verb}) : {bvs['mean_suppressed_revisions_per_sequence']:.1f}")
        if bvs["frac_committed_positions_where_belief_differs"] > 0.5:
            print(
                "  caveat: > 50% disagreement is at/above the 75% chance level for 4 tokens -- the\n"
                "  MDLM loss only supervises *masked* positions, so a model's logits at unmasked\n"
                "  positions can be untrained noise rather than a genuine wish to revise."
            )
    rq = summary.get("revision_quality") or {}
    if rq:
        print("\n[mind-change] === WERE THE REVISIONS GOOD? (FBCNN log p(class | seq)) ===")
        print(f"  events scored: {rq['n_events_scored']} over {rq['n_sequences']} sequences "
              f"(skipped, revision later undone: {rq['n_events_skipped_revision_undone']})")
        print(f"  CI95 = {rq['bootstrap']} -- events cluster by sequence, so a per-event "
              "binomial error bar is too narrow.")
        print(f"  {'':36}{'% improved (CI95)':>26}  {'mean delta (CI95)':>28}")
        for key, name in (
            ("revision", "revision vs its own old token"),
            ("null_random_token_same_position", "null: random token, same position"),
            ("null_random_position", "null: random token, random position"),
        ):
            b = rq[key]
            fl, fh = b["frac_improved_ci95"]
            ml, mh = b["mean_delta_ci95"]
            print(
                f"  {name:<36}{b['frac_improved'] * 100:7.2f} [{fl * 100:5.2f}, {fh * 100:5.2f}]"
                f"  {b['mean_delta']:+11.4f} [{ml:+.4f}, {mh:+.4f}]"
            )
        pr = rq["revision_vs_random_token_paired"]
        pl, ph = pr["frac_revision_better_ci95"]
        print(
            f"  paired (same event): revision beats a random alternative "
            f"{pr['frac_revision_better'] * 100:.2f}% [{pl * 100:.2f}, {ph * 100:.2f}]"
        )
        rev = rq["revision"]
        lo, hi = rev["frac_improved_ci95"]
        if lo <= 0.5 <= hi:
            print("  -> revisions are indistinguishable from a coin flip: the corrector is not correcting.")
        elif hi < 0.5:
            print("  -> revisions score *worse* than the token they replaced: the corrector is "
                  "actively degrading the sequence by this classifier's judgement.")
        elif pr["mean_delta_difference_ci95"][0] <= 0.0:
            print("  -> revisions beat their old token, but no better than an arbitrary alternative "
                  "at the same position: the model picks *where*, not *what*.")
        else:
            print("  -> revisions beat both their old token and an arbitrary alternative: real corrections.")
        # Sensitivity check: if a random single-base substitution barely moves the score, the
        # classifier cannot adjudicate single-position edits and none of the above means much.
        npos = rq["null_random_position"]
        if abs(npos["mean_delta"]) < 0.1 * npos["std_delta"]:
            print(
                f"  caveat: a random single-base substitution moves the score by only "
                f"{npos['mean_delta']:+.4f} nats (sd {npos['std_delta']:.4f}) -- this classifier "
                "barely responds to one-position edits, so the directions above can be resolvable "
                "yet negligible in magnitude. Confirm in bulk (FBD with revision on vs off) "
                "before concluding the revisions matter."
            )

    if summary["state"]["mean_switches_per_position"] == 0.0:
        if trainer != "routed_mdlm":
            print(
                "\n[mind-change] note: 0 state switches by construction -- the simple baseline is "
                "strict carry-over, so a committed token is frozen. This run is the control."
            )
        elif corruption_mode != "independent":
            print(
                "\n[mind-change] note: 0 state switches by construction -- corruption_mode="
                f"{corruption_mode!r} is strict carry-over. This run is the control."
            )
        elif not cli.free_support:
            print(
                "\n[mind-change] note: 0 state switches. With the support mask on, a committed "
                "token can only change by being re-masked first; re-run with --free_support to "
                "let the corrector phase overwrite committed tokens in place."
            )
    print(f"\n[mind-change] wrote {json_path}")
    print(f"[mind-change] wrote {csv_path}")
    print(f"[mind-change] wrote {pos_path}")


if __name__ == "__main__":
    main()
