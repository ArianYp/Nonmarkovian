# Computational-overhead ablation — SLM (Bernoulli-simplex) non-Markovian path

What does the router / history machinery actually cost, relative to the Markovian
SLM baseline, in parameters, FLOPs, wall-clock and memory — at training and at
sampling?

Harness: [`nonmarkovian/bench_overhead.py`](../nonmarkovian/bench_overhead.py).
Applies to the **SLM** path only (`RoutedDenoiserCNN` + vendored SLM `CNNModel`),
not the MDLM twin.

---

## 1. What is compared

| variant | model | input | role |
|---|---|---|---|
| `baseline` | `DiscreteDenoiserCNN` | `corrupt_sequence_bernoulli` → `[B, L, 4]` | Markovian control (= `train_simple.py` / `sample_simple.py`) |
| `routed@t_start=τ` | `RoutedDenoiserCNN` | `sample_all_views_bernoulli` → `[B, T−τ, L, 4]` | the method (= `train.py`), with `K = T − τ − 1` candidates |
| `routed_K0` | `RoutedDenoiserCNN` | same, at `τ = T−1` → `K = 0` | router isolator (see below) |
| `routed/trajectory` | `RoutedDenoiserCNN` | real reverse-process history | the method at inference |
| `routed/uniform` | `RoutedDenoiserCNN` | all candidate slots reset to `1/C` each step | compute-matched control |

Three design choices make the numbers defensible:

**The `routed_K0` isolator.** Instead of a separately-built stripped model
(different params, different weights, arguable), `routed_K0` is the *same routed
module* driven at `t_start = T−1`, which sends `forward` down the
`z_cand is None` branch ([`model.py:333`](../nonmarkovian/model.py#L333)). Router
and π-mixing are skipped; wrapper, parameters and denoiser are byte-identical.
So `routed@t_start=0 − routed_K0` is the router's cost and nothing else.

**Round-robin measurement.** One step of *every* variant per iteration, not each
variant looped to completion. A variant-by-variant loop charges whoever runs
first all of the allocator / autotune warmup — in an early version of this
harness that artifact reported the baseline as 2.5× *slower* than the routed
model.

**Analytic FLOPs alongside measured time.** `analytic_flops()` counts MACs in
closed form for both router branches (the `rk = 1` collapsed
`M = W_cur^T W_phi` path where `C_out` cancels, and the `rk > 1` conv path) and
for the CNN denoiser. Milliseconds are hardware-dependent; MACs are exact and
checkable. The gap between the two ratios is itself the main finding.

`--split_forward` additionally attributes the routed forward to *router* vs
*denoiser* via a synced `forward_pre_hook` on `model.cnn`.

---

## 2. How to run

```bash
# LSF, 1 GPU (~30 s for the configuration below)
python -m nonmarkovian.bench_overhead \
  --device cuda --batch_size 64 --seq_len 500 --num_timesteps 1000 \
  --cnn_stacks 4 --router_conv_kernel 1 --router_out_channels 256 \
  --t_start_frac 0.0 0.5 0.9 --split_forward \
  --sample_steps 10 --sample_batch 64 --iters 20 --warmup 5 \
  --csv logs/bench_overhead_slm.csv
```

No dataset and no checkpoint: synthetic `x0`, randomly initialised models, and
the real code paths (`forward.py` corruption, both model classes, both samplers,
the trainers' NLL, `train_timing`'s CUDA-synced clock). Random weights are valid
for timing/memory and meaningless for anything else.

Sweeps append to one CSV with `--csv_append --tag <label>`; useful axes are
`--num_timesteps`, `--router_conv_kernel`, `--router_out_channels`,
`--corruption_mode`.

---

## 3. Recorded run

LSF job 121039, `farm-gpu0510`, 2026-08-26, torch 2.3.1+cu121, 1 GPU.
B=64, L=500, T=1000, `cnn_stacks=4`, `rk=1`, `C_out=256`, `tau=0.01`,
`router_k=2`, `corruption_mode=independent`, `bernoulli_scheduler=loglinear`.
Training: 20 measured iters after 5 warmup. Sampling: 10 steps, batch 64,
3 runs after 1 warmup, 1 NFE/step (no CFG).

GPU model was not recorded in that run; the header now prints
`torch.cuda.get_device_name()`, so re-run if the exact device matters for
publication.

### 3.1 Parameters

| | value |
|---|---|
| routed total | 3,658,436 |
| baseline total | 3,666,116 |
| **router-only** (`W_cur`, `W_phi`, `state_router_proj`) | **2,560 = 0.070% of routed** |
| of which live in the current forward | 2,048 (`state_router_proj` is unused — `s_cur_router = None`) |

The routed/baseline *total* difference is not the router's cost:
`DiscreteDenoiserCNN` hardcodes `CNNModel(4, 81, …)` while `RoutedDenoiserCNN`
uses `num_labels or 1`, so the totals also differ by a ~10k `cls_embedder`.
**Quote the router-only count.**

### 3.2 Training step (ms, device-synced, mean of 20)

| variant | K | views | forward | router | denoiser | loss | backward | **step** | vs base | peak mem |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0 | 0.31 | 6.02 | — | — | 0.14 | 13.20 | **19.67** | 1.00× | 1314 MB |
| `routed_K0` | 0 | 0.34 | 6.24 | 0.13 | 6.11 | 0.15 | 13.19 | **19.92** | 1.01× | 1314 MB |
| `routed@900` | 99 | 0.56 | 6.72 | 0.77 | 5.96 | 0.15 | 13.43 | **20.87** | 1.06× | 1412 MB |
| `routed@500` | 499 | 1.73 | 7.10 | 1.14 | 5.96 | 0.15 | 13.71 | **22.69** | 1.15× | 1803 MB |
| `routed@0` | 999 | 3.18 | 7.91 | 1.95 | 5.96 | 0.15 | 14.03 | **25.27** | 1.28× | 2291 MB |

`ms_step_std` ≤ 0.02 ms for the routed rows, 0.10 ms for baseline.

### 3.3 Sampling (10 steps, batch 64, mean of 3)

| variant | ms/batch | ms/step | ms/seq | seq/s | peak mem |
|---|---|---|---|---|---|
| `baseline` | 69.66 ± 0.09 | 6.97 | 1.09 | 918.8 | 339.1 MB |
| `routed/trajectory` | 76.28 ± 0.24 | 7.63 | 1.19 | 839.0 | 347.3 MB |
| `routed/uniform` | 76.21 ± 0.06 | 7.62 | 1.19 | 839.8 | 347.3 MB |

**1.10× baseline wall-clock, +8.2 MB.**

### 3.4 Analytic FLOPs (MACs, forward, B=64, L=500, `cnn_stacks=4`)

Denoiser is 95.08 GMAC in every row.

| phase | K | `rk=1` | `rk=5` | `rk=9` |
|---|---|---|---|---|
| train, `t_start=0` | 999 | 0.26 GMAC (**0.27%**) | 172.2 GMAC (181%) | 303.2 GMAC (319%) |
| train, `t_start=500` | 499 | 0.13 GMAC (0.13%) | 86.1 GMAC (91%) | 151.6 GMAC (159%) |
| sample, 10 steps | 9 | ~0.00 GMAC (0.003%) | 1.71 GMAC (1.8%) | 3.02 GMAC (3.2%) |

At `rk = 1` the score collapses to `z_t^T M z_cand` with `M = W_cur^T W_phi`
(4×4), so **`C_out` cancels out entirely** — a wide router is free. At `rk > 1`
cost is linear in `K`, `C_out` *and* `rk`, and the router overtakes the denoiser.

---

## 4. Derived scaling laws

Fitted on the three routed rows; both hold to within measurement noise.

**Time is linear in K.** `(25.27 − 19.92) / 999` = **5.36 µs per candidate**
(B=64, L=500). Predicts 22.59 ms at K=499 (measured 22.69) and 20.45 ms at K=99
(measured 20.87).

**Peak memory = baseline + 2 × views buffer**, where buffer = `B(K+1)L·16` bytes:

| K | buffer | predicted | measured |
|---|---|---|---|
| 99 | 48.8 MB | 1412 MB | 1412 MB |
| 499 | 244.1 MB | 1802 MB | 1803 MB |
| 999 | 488.3 MB | 2290 MB | 2291 MB |

The `2×` is not the buffer stored twice — it is the **construction pipeline** in
[`forward.py:216-244`](../nonmarkovian/forward.py#L216-L244): in `independent`
mode `u`, `samples`, the pre-normalised `x_t` and the normalised copy are all
resident at once (four ×488 MB plus 122 MB bool masks). That phase, not backward,
sets the peak.

**Expected training slowdown is the K=499 row, 1.15× — not 1.28×.** `t_start` is
drawn uniformly ([`train.py:681`](../nonmarkovian/train.py#L681)) and cost is
linear in K, so the mean over `t_start` equals the mid-K row. Report 1.28× as
worst case.

---

## 5. Findings

**The overhead is memory bandwidth and kernel launches, not arithmetic.** The
router is 0.27% of denoiser FLOPs but +27.2% of a baseline step in wall-clock —
a ~100× gap. Same at sampling: 0.003% of FLOPs, 1.10× wall-clock. The views
construction step alone (3.18 ms) moves ~2.5 GB, i.e. it runs at roughly
780 GB/s — saturated. Consequence: the method is arithmetically free, and its
cost is an implementation property that can be attacked without touching the
math.

**Sampling is near-free at 10 steps, but that is a property of few-step
sampling.** The buffer is `B·steps·L·16` bytes: 4.9 MB at 10 steps, 49 MB at
100, 488 MB at 1000. Unlike training, there is no construction pipeline
(`_init_views_buffer` is a single `new_full`,
[`sample.py:98`](../nonmarkovian/sample.py#L98)) and no autograd, so the extra is
just the buffer plus one transient π-mixing product — 4.9 + 4.4 ≈ the +8.2 MB
observed. State the step-count dependence rather than "inference is free".

**`trajectory` and `uniform` are genuinely compute-matched**: 76.28 vs 76.21 ms
(inside one std) and identical peak memory to the decimal, because the uniform
reset is an in-place `fill_` ([`sample.py:229`](../nonmarkovian/sample.py#L229)).
Any quality gap between them is therefore pure history information, with no
compute or capacity confound.

**The router's candidate tensor is free.** `cand = x_views[:, t_start+1:T]` is a
slice (a view) and `.to(torch.float32)` on already-fp32 data returns the same
storage — no copy, at either phase.

### Internal consistency checks

- `ms_forward_denoiser` = 5.96 ms at every K — denoiser cost independent of
  candidate count, as it must be.
- `routed_K0` = 19.92 ms vs baseline 19.67 ms (1.01×) and identical peak memory —
  the isolator measures the router, not an architecture difference.
- Both scaling laws fit three points each to <1%.

---

## 6. Scope and caveats

- Compute only. No FBD, no NLL, no motif metrics — random weights make quality
  numbers meaningless here.
- `routed/uniform` is **not** the vanilla Markovian model: π-mixing still runs
  with a constant `1/C` candidate, so `z_t` is still softened by
  `w_cur`/`w_hist` ([`model.py:365`](../nonmarkovian/model.py#L365)). The vanilla
  control is the separate `baseline` row.
- `--split_forward` adds one device sync per routed forward, so it perturbs the
  routed total slightly (it is off by default).
- Single node, single GPU, no DDP. Multi-GPU changes the picture only through
  the memory ceiling, not the ratios.
- The `bernoulli_hat` history mode ([`sample.py:99`](../nonmarkovian/sample.py#L99))
  is legacy and out-of-distribution at step 1; it is not benchmarked. Add a
  fourth `SampleVariant` if it is needed.

### Companion quality ablations (not in this harness)

- `--history_mode uniform` at sampling and `--val_no_history` at training:
  same FLOPs, same params, zero history information. Given the FLOP overhead is
  ~0.3%, this is the ablation that answers "is the gain just extra compute?".
- Wall-clock-matched baseline: train `train_simple.py` for
  `steps × (t_routed / t_baseline)` using the ratios in §3.2.
- End-to-end per-epoch timings on the real data path already exist via
  `--log_timing` in both trainers (`train/epoch_time_ms_*` in W&B).

---

## 7. Open knobs, in order of expected payoff

1. **`--corruption_mode trajectory`** — draws `u` as `[B,1,L,4]` instead of
   `[B,K,L,4]` ([`forward.py:230`](../nonmarkovian/forward.py#L230)), removing one
   full-size tensor and one 488 MB RNG pass from the construction peak. Should
   cut both the 3.18 ms views step and ~490 MB of peak memory. One flag, one
   re-run. Current runs use `independent`.
2. **Candidate window** — cost is O(K); restricting candidates to the last W
   states makes it O(W) and collapses the overhead toward 1.0×. The
   `train/router_t_start_minus_argmax_t_*` logs already indicate how far back the
   router actually reaches. Note `--router_k` saves nothing: it prunes *after*
   scoring, and scoring is the cost.
3. **bf16 views buffer** — halves the bandwidth and the `2 × buffer` memory term;
   at `rk=1` the router math is a 4×4 einsum that can accumulate in fp32.

---

## 8. Artifacts

- Harness: [`nonmarkovian/bench_overhead.py`](../nonmarkovian/bench_overhead.py)
- Raw CSV: `logs/bench_overhead_slm.csv` (one row per variant; config columns
  `tag`, `device`, `batch_size`, `seq_len`, `num_timesteps`, `cnn_stacks`,
  `router_conv_kernel`, `router_out_channels`, `sample_steps`, `sample_batch`,
  `params_*`, then per-variant `ms_*`, `peak_mem_mb`, `views_buffer_mb`,
  `router_macs`, `denoiser_macs`, `router_frac_pct`)
- Job log: `logs/lsf-distill-vanilla-121039.out`
