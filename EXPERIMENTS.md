# Non-Markovian discrete diffusion — two evaluation experiments

Notes for co-authors. Covers what each experiment asks, how it is implemented, what the current
numbers say, and what is still open. Fly-brain enhancer data (81 classes, L = 500) unless stated.

---

## Why there are two experiments

The paper's claim is that a **non-Markovian** reverse process helps because the model can *revise*
decisions it has already made. That needs two separate pieces of evidence:

1. **Does the model actually revise?** — a mechanistic measurement of the sampler's trajectory.
   This is the *mind-change* experiment.
2. **Do the resulting sequences look more like real enhancers?** — a quality measurement on the
   output. This is the *motif* experiment.

The second experiment exists in its current form because our learned scorer turned out to be
untrustworthy for the first (see [The scorer problem](#the-scorer-problem)).

---

## Experiment 1 — Mind-change

**Code**

| file | role |
|---|---|
| `nonmarkovian/mind_change_core.py` | backbone-agnostic metrics; imports only `torch` |
| `nonmarkovian/mind_change_slm.py` | SLM / ShortListing driver → `logs/mind_change/slm_fb_*` |
| `nonmarkovian/mind_change_mdlm.py` | MDLM driver → `logs/mind_change/nm_*` |

Both drivers import the same metric code, so the two model families report numbers that mean the
same thing.

### What counts as "changing its mind"

The shared interface is a **state trajectory** `[B, F, L]` of token ids, with `UNDECIDED = 4`
marking a position the sampler has not committed to. Each backbone supplies its own notion of
committed:

- **MDLM** — the position is unmasked (not `[M]`).
- **SLM** — the position's shortlist has collapsed to a single base, i.e. `x_t` is one-hot there.

A **switch** is a change between consecutive *committed* values at a position, bridging undecided
frames: `C C [?] C G` counts as one switch (C → G), regardless of how many undecided frames sit
between.

Two trajectories are recorded per run, and the distinction matters:

- **state** — what the sampler holds (the shortlist collapsed to committed / undecided).
- **belief** — `argmax` of the model's predicted clean sequence at each step, read *before* the
  support mask. What the model would say if it were allowed to act.

### Mechanism: how an SLM switch can happen at all

In `sample.py`, the reverse step has two regimes, gated at
`i > threshold * num_steps // 10`:

- **Markov regime** (`corruption_mode='trajectory'`, or early steps) —
  `_sample_bernoulli(predicted) & (x_t > 0)`. The active set can only *shrink*, so a collapsed
  position is frozen and switch counts are **0 by construction**. This is the control.
- **Non-Markov regime** (`corruption_mode='independent'`, past the gate) — the `& (x_t > 0)`
  intersection is dropped, so classes can **re-activate**. A collapsed position can re-expand and
  later collapse onto a *different* base.

Because the logits are masked to the current support, a committed position has
`predicted = 1.0` for the base it holds, so that base always survives the Bernoulli draw. Therefore
a switch can **never** happen between adjacent frames — it must route through a re-expanded
intermediate:

```
one-hot A  →  {A, C}  →  one-hot C      the only available route
one-hot A  →  one-hot C                 impossible in one step
```

The data confirms this exactly: `mean_switches_adjacent_per_position = 0.0`, and
`42470 refill events × (1 − 0.8364) = 6949`, which equals the `6929 scored + 20 skipped` revision
events. **Every** state-level switch is a re-expand-then-collapse-elsewhere event; nothing else
produces one.

### Two ways to measure revision

This is the most important conceptual point in the experiment.

| | `MindChangeStats` | `FinalBaseExclusionStats` |
|---|---|---|
| reads | the collapsed state trajectory | the raw shortlist bitmasks + the final sequence |
| asks | "did the *committed* base change?" | "was the base finally chosen ever *missing* from the shortlist?" |
| needs a commitment | **two** | **none** |
| available in | MDLM and SLM | **SLM only** |
| measured | **5.4%** of positions | **28.7%** of positions |

`FinalBaseExclusionStats` is strictly stronger. A position can rule the eventual answer out and come
back to it *without ever committing to anything else* — invisible to switch counting:

```
frame 0   {A,C,G,T}    state = UNDECIDED
frame 1   {A,C,T}      G ruled out       ← exclusion, but no switch
frame 2   {A,C}        G still out       ← exclusion, but no switch
frame 3   {A,C,G}      G re-admitted
frame 4   {G}          commits to G
final     G
```

`MindChangeStats` sees one commitment and reports **0 switches**. `FinalBaseExclusionStats` reports
2 excluded steps. That pattern is most of the 5.4% → 28.7% gap.

MDLM has no analogue because its state is binary — a token or `[M]`, with no partial candidate set.

### Results — SLM, fly brain

`run-20260704_054002-xmq8gvs7/routed.best_fbd.pt`, 256 sequences, T = 10,
`corruption_mode=independent`, guidance 0.6, conditional, support mask on.

**State trajectory (what the sampler holds)**

| quantity | value |
|---|---|
| switches per position | 0.054 |
| switches per sequence | 27.1 |
| positions ever changed | **5.39%** |
| max switches at one position | 2 |
| adjacent-frame flips | **0.0** |
| support re-expansions per position | 0.332 |
| re-expand → re-collapse events | 42,470, of which **83.6%** return the same base |
| **final base ever ruled out earlier** | **28.7% of positions, in 100% of sequences** |

The 4×4 from→to switch matrix is essentially uniform — revisions show no base-pair preference.

**Belief trajectory (what the model wants)**

2.95 switches per position, 94.7% of positions ever change. The model's opinion churns constantly
while the sampler commits to almost none of it.

**Belief vs committed state**

`disagree_per_step = 0.0` at every step. With the support mask on, the model *never* holds an
opinion contradicting an already-committed position — masking the logits pins a collapsed position
to the base it holds. `--free_support` lifts this. **That arm has not been run.**

### Results — MDLM

`run-20260811_151404-bl11w6g0/routed_mdlm_mel.best.pt`, T = 32, guidance 0.5, threshold 0.6.
Note this is the **melanoma** dataset, not fly brain.

| quantity | MDLM | SLM |
|---|---|---|
| switches per position | 0.105 | 0.054 |
| positions ever changed | 10.1% | 5.4% |
| re-masks per position | 1.126 | 0.332 |
| refills returning the same token | 90.7% | 83.6% |
| max switches at one position | 3 | 2 |

Not a clean head-to-head: different dataset **and** different step count.

### Was the revision any good?

Two independent judges, both using the FBCNN classifier (`fbd.ckpt`).

**Per-event, class log-probability** (`RevisionScorer`, clustered bootstrap over sequences):

| contrast | % improved | CI95 |
|---|---|---|
| revision vs its own old base | 48.8% | [47.5, 50.3] |
| null: random base, same position | 48.6% | [47.0, 50.1] |
| null: random base, random position | 49.4% | [47.9, 51.0] |
| paired: revision beats a random alternative | 50.3% | [48.9, 51.6] |

Indistinguishable from a coin flip. **But this is criterion insensitivity, not evidence the
corrector is useless**: the random-position null moves the score by −0.003 nats against a standard
deviation of 0.91, i.e. the classifier barely responds to single-base edits at all.

**Distributional, FBD** (`RecoveryFBD`, 143 positions reverted per sequence, paired bootstrap):

| set | FBD to real | Δ vs actual | CI95 |
|---|---|---|---|
| actual output | 3.10 | — | — |
| revert recoveries to best surviving candidate | 9.55 | **+5.88** | [−2.41, 15.62] |
| revert to a random surviving candidate | 31.59 | +27.40 | [16.33, 36.25] |
| null: same count, random positions | 29.50 | +26.19 | [16.46, 37.98] |

Lower FBD = closer to real, so Δ > 0 means the corrector helped. The **null registers cleanly**
(+26.19, CI excludes 0), so the measurement works. The result of interest, +5.88, has 92% of
bootstrap draws positive but a CI crossing zero — **suggestive, underpowered at n = 256.**

The switch-level arm (revert to first commitment) is **not readable**: its null
(`random_positions_switch`) *lowered* FBD, i.e. random damage improved the score. Only ~27 positions
per sequence are perturbed there, below FBD's noise floor at this sample size. The script detects
this and refuses the comparison.

### Known issues — mind-change

1. **`threshold = 3` in `sample.py` is uncommitted local state, and the docstrings say 6.** The gate
   is a *fraction* — `threshold/10` of the steps — so 3 means re-noising starts 30% of the way
   through, not 60%.
2. **The re-noise branch wipes the state twice.** It uses `weight = clamp(nominator, 0, 1)`, dropping
   the `/denominator` the Markov branch has. `nominator = 4^(t − 1/T) − 1` saturates at 1.0 for
   t ≥ 0.5, making `predicted ≡ 1` and re-activating all four bases everywhere. The trace shows it:
   `frac_undecided_per_step = [1.0, .99, .96, .90, 1.0, 1.0, .977, .87, .68, .42, .04, 0]` — frames
   4 and 5 are full resets to uniform. Every measured mind-change therefore happens in the last ~4
   of 10 steps. **Decide whether this is intended before publishing.**
3. **`threshold` is hardcoded** with debug `print`s around it, not a CLI flag. It needs to become an
   argument for reproducibility, and it would enable the threshold sweep below.
4. **`--free_support` never run** — the direct test of the suppressed revisions.
5. **`--corruption_mode trajectory` never run for SLM** — should give 0 switches by construction.
6. **T = 10 was used** although the checkpoint stores `num_timesteps_sample = 50`.

---

## The scorer problem

Worth stating explicitly in the paper, because it motivates Experiment 2.

The FBCNN classifier **cannot be trusted per sequence** on fly brain:

- its class head puts real held-out sequences **below chance** under their own labels;
- embedding proximity (Mahalanobis, kNN) ranks **randomised** real sequences *above* real ones,
  because noise drags a mean-pooled CNN embedding toward the dataset centroid;
- a single-base substitution moves the class log-probability by −0.003 nats against sd 0.91.

Only the **distributional** use (FBD) survives a positive control. This is why the per-sequence
criteria in `mind_change_slm.py` are kept but flagged, and why `motif_metrics.py` was written.

Both scripts share a design rule worth keeping: **every reported verdict first checks whether its
null registered.** Several verdicts in the output read "this measurement cannot be read", which is
the intended behaviour.

---

## Experiment 2 — Motif content

**Code**: `nonmarkovian/motif_metrics.py`. No learned scorer — a PWM match either clears its
threshold or it does not, so the measurement is interpretable at any sample size and its positive
control is trivial to verify.

### How the measurement works

1. **Parse** HOMER `.motif` files: header is `>consensus⇥name⇥threshold`, then a position weight
   matrix. Using HOMER's own per-motif threshold avoids inventing a cutoff.
2. **Background** = the observed base composition of the *real* sequences (`[27,23,23,27]%` here),
   so an AT-rich genome cannot inflate AT-rich motifs.
3. **Score** — each PWM becomes a log-odds matrix, `lo[i][c] = log2(p[i][c] / bg[c])`, applied as one
   `conv1d` with a 2-channel kernel (forward + reverse complement). Every offset gets a score in
   bits; `max` is taken over strands *before* thresholding, so a palindromic site occupied on both
   strands counts once, not twice.
4. **Hit** = `score >= threshold`. For a 500 bp sequence and 78 motifs that is ~76,000 scored
   offsets per sequence.
5. **Floor** — a **dinucleotide-shuffled** copy of the real data is scored automatically. It
   preserves base *and* dinucleotide frequencies while destroying motif order, so it shows how much
   of any method's motif content is explained by composition alone.

### What the two headline metrics mean

For each of the 78 motifs you get one number per set: mean hits per sequence. Comparing a set to
real is comparing two lists of 78 numbers, and the two metrics ask different questions.

**`mean |log2FC|` — are the numbers right?** Per motif: ratio = generated ÷ real, take `log2`
(so 2× and ½× are symmetric at +1 and −1), take the absolute value (so over- and under-production
do not cancel), then average over the 78 motifs. Lower is better; 0 is perfect.
`0.144` means `2^0.144 = 1.105`, i.e. **the typical motif is ~10% off**.

**`pearson` — is the pattern right?** Correlation between the two 78-number lists. Where real uses a
motif heavily, does the model too? It is **blind to overall scale**.

Neither suffices alone. Tested on the real profile:

| hypothetical model | pearson | mean\|log2FC\| | hits ratio |
|---|---|---|---|
| produces exactly **half** of real, every motif | **+1.000** | 1.000 | 0.500 |
| right **total**, but motifs scrambled | −0.106 | 1.305 | **1.000** |

The halving model gets a *perfect* pearson while every number is wrong by 2×. The scrambled model
gets a *perfect* hits ratio while using entirely the wrong motifs.

### Results

`data_dfm/the_code/Fly/data/homer/M0_vs_M10/knownResults`, 78 motifs, real test split, n = 8192 per
set.

**Raw counts**

| set | total hits/seq | motifs present/seq |
|---|---|---|
| real | 22.60 | 15.28 |
| real, dinucleotide-shuffled | 26.04 | 17.59 |
| nonmarkov | 21.46 | 14.78 |
| markov | 21.12 | 14.49 |
| simple | 21.54 | 14.82 |

**Profile agreement with real**

| set | pearson ↑ | mean\|log2FC\| ↓ | typical per-motif error | spearman | hits ratio | dep2× | enr2× |
|---|---|---|---|---|---|---|---|
| **nonmarkov** | **0.991** | **0.144** | ≈10% | 0.990 | 0.950 | 0 | 0 |
| markov | 0.984 | 0.176 | ≈13% | 0.986 | 0.934 | 0 | 0 |
| simple | 0.963 | 0.214 | ≈16% | 0.981 | 0.953 | 0 | 0 |
| *shuffled real (floor)* | *0.926* | *0.329* | *≈26%* | *0.965* | *1.152* | *0* | *1* |

A clean monotone ordering — **nonmarkov > markov > simple > floor** — on both headline metrics. Two
findings:

1. Both routed models beat the simple baseline (0.963 → 0.984 / 0.991): the routed/history
   architecture is doing work.
2. Non-Markov beats Markov (0.984 → 0.991): the paper's claim.

`dep2× = enr2× = 0` for all three models — not one of the 78 motifs is off by more than 2× in either
direction.

The ordering is **robust to scale choice**: recomputing pearson on log-transformed profiles gives
0.993 / 0.987 / 0.978 / 0.959, the same ordering. The profile is also not tail-dominated (real
hits/seq spans 1.158 → 0.047, median 0.20; the top-10 motifs carry only 36% of all hits), so pearson
is not being carried by a handful of frequent motifs.

That the ordering is monotone across three arms and two metrics simultaneously is worth more than
any single pairwise gap.

### How to state the claim

**Supported**: all three models reproduce the real motif-frequency *profile* far better than a
composition-matched scramble does, and the non-Markovian model reproduces it best.

**Not supported**: "the models generate more motifs." They generate slightly *fewer* than real
(ratio 0.950 / 0.934 / 0.953), and the shuffle generates *more* (1.152). Frame this as **profile
agreement**, never motif abundance.

Report `mean |log2FC|` as primary with `pearson` beside it, and **always print the floor row** — it
gives the reader the scale and turns three decimals near 1.0 into a measurement.

### Columns to demote

- **`spearman`** — range across *all* rows is 0.965–0.990; the floor sits within 0.025 of the best
  model. Rank order of motif frequencies is easy to get roughly right, so it barely discriminates.
- **`hits ratio`** — one degree of freedom, and it *contradicts* the ordering (simple 0.953 ≈
  nonmarkov 0.950, markov worst at 0.934). Sanity check only.
- **`dep2×` / `enr2×`** — all zero for every model. One sentence, not a column.

### Validation and known issues — motif

**The scanner is validated against HOMER.** HOMER reported de novo `motif1`'s background rate as
`B: 37.62%`; our scan of real class-10 sequences gives **37.6%**. Exact agreement — the log-odds,
thresholds, and strand handling are HOMER-compatible, and "M10" is class 10.

**The floor produces more raw hits than real, and that is expected here.** Real fly enhancers are
*depleted* in these motifs relative to a dinucleotide-matched null (for de novo `motif1`: 32.6% real
vs 51.2% shuffled). The reason is motif degeneracy — `NNNNGHGATCDY` is effectively "GATC" with four
wildcards, and its occurrence rate is largely determined by dinucleotide frequencies, which the
shuffle preserves *exactly*. Raising the threshold does not change this: sweeping
`--threshold_scale` 1.0 → 1.4 drops total hits from 22.3 to 2.1 while the hits ratio only moves
1.169 → 1.014, never below 1. **This is why the comparison must be framed as profile agreement**,
which does separate the methods from the floor cleanly.

**Open issues**

1. **No error bars.** The margin carrying the headline claim (0.984 → 0.991) is the *smallest* gap in
   the table. `compare()` computes point estimates only. A paired bootstrap over the 8192 sequences —
   shared resample indices across sets, as `RecoveryFBD` already does — would settle whether it is
   resolvable. This is the single most reviewer-exposed gap.
2. **The motif library is not Drosophila.** `knownResults/` is HOMER's generic cross-species known
   set: Arabidopsis DAP-Seq (`col-GATA14`, `colamp-SPL15`, ANAC/VND), mouse/human ChIP
   (`Neuron-Isl1`, `NPC-Sox3`, `Myotubes-Sox6`), chicken Pitx1. These are the entries HOMER found
   *enriched* in the fly comparison, so they are sequence-similar to real signal, but a reviewer will
   notice. 28 **de novo** motifs discovered in this dataset sit in `homerResults/` and should be run
   as a robustness check (the job costs ~11 s).
3. **Arm matching unverified.** `nonmarkov` and `markov` share a checkpoint and differ only in
   `corruption_mode`; `simple` is a different checkpoint. Confirm `guidance_scale` and step count
   agree across all three, or the comparison carries a second variable.
4. **Markov arm is an inference-time ablation, not a Markov-trained baseline.** The checkpoint's own
   config says `corruption_mode = independent`, and `corruption_mode` in `sample.py` is inference-only.
   So this arm answers *"does the model use the revision ability it was trained for?"*, not *"is the
   non-Markovian objective better?"* Both are legitimate; be explicit about which. Markov-*trained*
   fly-brain checkpoints exist (`run-20260506_115626-0kx20yhi`, `run-20260506_120133-b8pl04jn`) but
   are from May 6 vs July 4, so diff their configs before using either as a baseline.
5. **`read_sequences` was FASTA-only** and silently collapsed a plain one-sequence-per-line file into
   a single record, so an earlier run reported `n = 1` for every generated set. Fixed; all numbers in
   this document are post-fix.
6. **`motif_metrics.py` docstring points at `data_motifs/`, which is empty.** The real path is
   `data_dfm/the_code/Fly/data/homer/M0_vs_M10/knownResults`.
7. **`parse_homer_motif` reads only the first record per file.** A multi-motif file such as
   `homerMotifs.all.motifs` would be merged into one bogus concatenated PWM. Use the per-file
   `homerResults/*.motif` route.

---

## Reproduction

### Mind-change

```bash
CKPT=wandb/run-20260704_054002-xmq8gvs7/files/routed.best_fbd.pt

# non-Markovian (the method)
python -m nonmarkovian.mind_change_slm --checkpoint $CKPT \
    --corruption_mode independent --guidance_scale 0.6 \
    --n_samples 256 --num_timesteps_sample 10 --split val \
    --score_revisions --score_recoveries --fbcnn_ckpt fbd.ckpt \
    --recovery_metric fbd --out_dir logs/mind_change --tag slm_fb

# Markovian control — expect 0 switches by construction
python -m nonmarkovian.mind_change_slm --checkpoint $CKPT \
    --corruption_mode trajectory --guidance_scale 0.6 \
    --n_samples 256 --num_timesteps_sample 10 --tag slm_fb_markov
```

`--tag` is the output filename prefix; without it, successive runs overwrite each other.

### Sequence dumps for the motif experiment

`--dump_sequences` draws labels from the split, so the generated class mix matches the real data.
Preferred over `sample.py --out`, which has no `--guidance_scale` or `--n_samples`.

```bash
python -m nonmarkovian.mind_change_slm --checkpoint $CKPT \
    --corruption_mode independent --guidance_scale 0.6 \
    --n_samples 8192 --batch 64 --split test \
    --dump_sequences logs/motifs/seqs_nonmarkov.txt --tag nm_dump
```

Each dump run also produces a full mind-change stats file for free.

### Motif comparison

```bash
python -m nonmarkovian.motif_metrics \
    --motif_dir data_dfm/the_code/Fly/data/homer/M0_vs_M10/knownResults \
    --real_split test --no_dfm_melanoma --n_real 8192 --max_len 500 \
    --sets nonmarkov=logs/motifs/seqs_nonmarkov.txt \
           markov=logs/motifs/seqs_markov.txt \
           simple=logs/motifs/simple.txt \
    --out_dir logs/motifs --tag fly
```

Outputs `logs/motifs/fly_motif_metrics.json` and `fly_motif_per_motif.csv` (per-motif `hits_per_seq`
and `hit_rate` for every set). Runs in ~11 s on CPU.

---

### MDLM setting — fly brain (Sept 2026 runs)

Everything above was measured on the **SLM** backbone. The Sept 16–17 MDLM runs are a different
setting and need the MDLM driver, `mind_change_mdlm.py`. Shared properties of all four
checkpoints: CNN backbone (4 stacks), `max_len 500`, `num_timesteps 1000`,
`num_timesteps_sample 10`, and — for the routed ones — `corruption_mode=independent` with
**`independent_threshold = 0`**, i.e. the non-Markovian regime is on from the first reverse step
(the SLM runs gated it at 0.6).

**Sampling settings are not the checkpoint defaults.** The `eval_checkpoint_mdlm` runs that report
FBD ≈ 3–5 on the fly routed model override them: `--num_timesteps_sample 5 --guidance_scale 1
--independent_threshold 0 --split test` (a guidance sweep is in `logs/eval-mdlm-1542*.out`:
w = 1 → 3.84, w = 2 → 4.55, w = 3 → 8.62). Training-time FBD selection ran at guidance 0, so
`best_fbd.pt` was *selected* at 0 but is *evaluated* at 1. Every mind-change and motif run must use
the same three overrides, or its numbers are not comparable to the reported FBD.

| arm | checkpoint |
|---|---|
| routed, conditional (81 cls) | `wandb/run-20260916_155923-qw9q9xgj/files/routed_mdlm.best_fbd.pt` |
| simple, conditional (81 cls) | `wandb/run-20260917_133620-jilygebj/files/simple_mdlm.best_fbd.pt` |
| routed, unconditional | `wandb/run-20260916_144601-452z2a31/files/routed_mdlm.best_fbd.pt` |
| simple, unconditional | `wandb/run-20260917_133624-dvc41u7u/files/simple_mdlm.best_fbd.pt` |

Needs a GPU node (see `Nonmarkovian.slurm`); the login node has none.

#### Conditional

```bash
NM=wandb/run-20260916_155923-qw9q9xgj/files/routed_mdlm.best_fbd.pt
SIMPLE=wandb/run-20260917_133620-jilygebj/files/simple_mdlm.best_fbd.pt

# non-Markovian (the method) — the only arm that can score revisions
python -m nonmarkovian.mind_change_mdlm --checkpoint $NM \
    --corruption_mode independent --guidance_scale 1 \
    --num_timesteps_sample 5 --independent_threshold 0 \
    --n_samples 256 --batch 64 --split test \
    --score_revisions --fbcnn_ckpt fbd.ckpt \
    --out_dir logs/mind_change --tag mdlm_fb_nm

# Markovian control — same checkpoint, expect 0 state switches by construction
python -m nonmarkovian.mind_change_mdlm --checkpoint $NM \
    --corruption_mode trajectory --guidance_scale 1 \
    --num_timesteps_sample 5 \
    --n_samples 256 --batch 64 --split test --tag mdlm_fb_mk

# simple baseline — strict carry-over, also 0 switches; read its `belief` block
python -m nonmarkovian.mind_change_mdlm --checkpoint $SIMPLE \
    --guidance_scale 1 --num_timesteps_sample 5 --n_samples 256 --batch 64 --split test --tag mdlm_fb_simple

# optional: the suppressed-revision arm (routed only)
python -m nonmarkovian.mind_change_mdlm --checkpoint $NM \
    --corruption_mode independent --guidance_scale 1 --free_support \
    --num_timesteps_sample 5 --independent_threshold 0 \
    --n_samples 256 --batch 64 --split test --tag mdlm_fb_nm_free
```

#### Unconditional

Same commands with `$NM`/`$SIMPLE` pointing at the `no_labels` checkpoints. Drop
`--score_revisions`: it scores `log p(target class | seq)` and exits with an error on a checkpoint
that has no classes. `--split` is then only used by `motif_metrics` for the real reference profile.

```bash
NM=wandb/run-20260916_144601-452z2a31/files/routed_mdlm.best_fbd.pt
SIMPLE=wandb/run-20260917_133624-dvc41u7u/files/simple_mdlm.best_fbd.pt
```

#### Use the MDLM driver, not the SLM one

`mind_change_slm.py` on an MDLM checkpoint used to run **silently** and produce nonsense: the two
families share the architecture, so the weights load, and sampling then goes through the Bernoulli
ShortListing reverse process. The symptom is FBD ≈ 80 where `eval_checkpoint_mdlm` reports ≈ 3–5,
with the recovery nulls inverted (reverting revisions "improves" FBD, and the script prints
*"the null does not raise FBD, so this comparison cannot be read"*). Both trainer detectors now
refuse a checkpoint from the other family with an error naming the right script.

`mind_change_mdlm.py` now has the distributional arm too: `--score_revision_fbd` (alias
`--score_recoveries`). It reverts every switched position to its first commitment and compares
`FBD(real, reverted)` against `FBD(real, actual)`, with the count-matched random-position null and
a random-third-token arm, over a paired bootstrap (`--fbd_boot`, default 200; reference cloud
`--n_ref`, default 2048). **Lower FBD is better, so a positive delta means reverting made it worse,
i.e. the revisions helped.** Unlike `--score_revisions` it needs no labels.

The shortlist-recovery arms (`best_surviving` / `random_surviving`) have no MDLM analogue — they
read the shortlist bitmasks, and MDLM's state is binary — so `--score_recoveries` on the MDLM
driver runs the revision arms only. `--recovery_metric` and the per-sequence
`RecoveryCounterfactualScorer` remain SLM-only; the per-sequence classifier criterion is the one
that fails its positive control anyway.

**Read the null first.** If `null: same count, random positions` does not *raise* FBD, the run is
unreadable and the script says so — random damage scoring better than the model means the
generated set is far enough from real that any perturbation drags it toward the centroid. That is
what the FBD ≈ 80 run showed. It also needs sequences: at `--n_samples 64` the null fails purely on
sample size, so keep `--n_samples 8192`.

#### Sequence dumps for the motif experiment

`mind_change_mdlm.py` gained `--dump_sequences` (Sept 2026), matching the SLM driver: on a
conditional checkpoint the labels come from `--split`, so the generated class mix matches real.
`sample_mdlm.py --out` is **not** a substitute — it takes one fixed `--label` for the whole run.

`eval_checkpoint_mdlm.py` also gained `--dump_sequences` (plus `--dump_real` and `--n_dump`), the
same flags `eval_checkpoint.py` has. Use that route when you want the FBD number and the motif
dump from one run: the dump reuses the FBD pass's own samples, so the file is exactly the set the
printed FBD scored, and `--fbd_no_history` writes its uniform-history pass to
`<stem>_no_history.txt`. Use the `mind_change_mdlm.py` route when you want the mind-change stats
alongside the dump. The two are not interchangeable: they seed their samplers differently, so
pick one route for all arms of a comparison.

```bash
python -m nonmarkovian.eval_checkpoint_mdlm --checkpoint $NM \
    --split test --fbcnn_ckpt fbd.ckpt --guidance_scale 1 \
    --num_timesteps_sample 5 --independent_threshold 0 \
    --dump_sequences logs/motifs/mdlm_fb_nonmarkov.txt \
    --dump_real logs/motifs/real_fb_test.txt
```

```bash
for arm in nonmarkov markov; do
  [ $arm = nonmarkov ] && MODE=independent || MODE=trajectory
  python -m nonmarkovian.mind_change_mdlm --checkpoint $NM \
      --corruption_mode $MODE --guidance_scale 1 \
      --num_timesteps_sample 5 --independent_threshold 0 \
      --n_samples 8192 --batch 256 --split test \
      --dump_sequences logs/motifs/mdlm_fb_$arm.txt --tag mdlm_fb_${arm}_dump
done

python -m nonmarkovian.mind_change_mdlm --checkpoint $SIMPLE \
    --guidance_scale 1 --num_timesteps_sample 5 --n_samples 8192 --batch 256 --split test \
    --dump_sequences logs/motifs/mdlm_fb_simple.txt --tag mdlm_fb_simple_dump
```

#### Motif comparison

```bash
python -m nonmarkovian.motif_metrics \
    --motif_dir data_dfm/the_code/Fly/data/homer/M0_vs_M10/knownResults \
    --real_split test --no_dfm_melanoma --n_real 8192 --max_len 500 \
    --sets nonmarkov=logs/motifs/mdlm_fb_nonmarkov.txt \
           markov=logs/motifs/mdlm_fb_markov.txt \
           simple=logs/motifs/mdlm_fb_simple.txt \
    --out_dir logs/motifs --tag mdlm_fly
```

Do not mix a dump from a conditional checkpoint with one from an unconditional checkpoint in the
same `--sets` call: the class mix differs, which is a second variable on top of the arm.

## Suggested next steps, in priority order

1. **Paired bootstrap on the motif comparison** — the headline margin has no error bar. Cheap, and
   the most exposed gap.
2. **Resolve `threshold = 3`** — reconcile with the docstrings, decide whether the double state reset
   is intended, and promote it to a CLI flag.
3. **Run the two missing mind-change arms** — `--corruption_mode trajectory` (the 0-switch control)
   and `--free_support` (the suppressed-revision test).
4. **De novo motif library** as a robustness check on the main result.
5. **Threshold sweep as an ablation** — `threshold ∈ {10 (≡ Markov), 6, 3, 1}` against switch counts
   and motif quality answers "how much revision budget does the model need?". Note `threshold ≥ 10`
   is bit-identical to `corruption_mode=trajectory`, which is a free consistency check.
6. **Decide the Markov baseline story** — inference-time ablation, Markov-trained baseline, or both.
7. **Matched MDLM/SLM comparison** — currently confounded by dataset and step count.

## Cross-experiment note

The two experiments agree, which is the strongest version of the argument. The mind-change FBD arm
is suggestive but underpowered (+5.88, CI crossing zero); the motif metric — with no learned
component anywhere in it — puts the non-Markovian model first on both headline columns. Two
independent measurements pointing the same way is a substantially better paper than either alone.
