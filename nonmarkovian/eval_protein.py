"""Foldability + fitness evaluation for protein checkpoints (EvoDiff-style metrics).

Generates sequences from a trained checkpoint (routed *or* simple baseline -- auto-detected
from the checkpoint's ``trainer`` field) and scores them with the two standard
protein-design metrics used by EvoDiff / SLM-protein papers:

- **Fitness / naturalness**: ESM-2 **pseudo-perplexity** (masked-marginal). Lower = the
  sequence is more "natural" under a large protein LM. (``--esm2_model``, default 650M.)
  Optionally also **ProGen2-NLL** (``--progen2``): a *true* autoregressive negative
  log-likelihood (nats/residue) under the causal ProGen2 LM, complementing ESM-2's
  bidirectional pseudo-likelihood. (``--progen2_model``, default ``hugohrban/progen2-xlarge``.)
- **Foldability**: **ESMFold mean pLDDT** (0-100). Higher = more confidently foldable;
  EvoDiff reports mean pLDDT and the fraction with pLDDT > 70. (``--esmfold``.)

Both heads are optional and lazily imported (ESMFold weights are large and need a GPU).
You can also skip generation and score an existing FASTA / one-per-line file via
``--sequences``.

Examples
--------
    # generate 100 seqs of length 200 from a routed checkpoint and score both metrics
    python -m nonmarkovian.eval_protein --checkpoint checkpoints/protein_routed.best.pt \\
        --num_seqs 100 --seq_len 200 --esmfold

    # fitness only (no ESMFold), comparing the baseline checkpoint
    python -m nonmarkovian.eval_protein --checkpoint checkpoints/protein_simple.best.pt \\
        --num_seqs 100 --seq_len 200 --no_esmfold

    # score sequences from a file produced by sample_protein.py
    python -m nonmarkovian.eval_protein --sequences protein_samples.txt --esmfold
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.vocab_protein import CAN_AAS, STOP, decode

_CANON = set(CAN_AAS)  # 20 standard amino acids


def _postprocess(seq: str) -> str:
    """SLM generation post-processing: truncate at the first STOP ('*') token."""
    return seq.split(STOP)[0]


def _clean(seq: str) -> str:
    """Truncate at STOP, then keep only the 20 canonical AAs (drop gap/specials/ambiguous)."""
    return "".join(c for c in _postprocess(seq).upper() if c in _CANON)


def load_test_sequences(uniref: str, n: int, max_len: int, seed: int) -> list[str]:
    """Draw ``n`` natural sequences from the UniRef **test** split (reference baseline)."""
    import numpy as np

    from nonmarkovian.data_protein import UniRefDataset, resolve_uniref_root

    root = resolve_uniref_root(uniref)
    ds = UniRefDataset(root, "test", max_len=max_len)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    return [ds[int(i)][0] for i in idx]


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_sequences(args, device) -> list[str]:
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})
    trainer = str(ckpt.get("trainer", ""))
    kind = args.model
    if kind == "auto":
        kind = "simple" if "simple" in trainer else "routed"
    num_steps = args.num_steps if args.num_steps > 0 else int(cfg.get("num_timesteps", 10))
    if args.rank0:
        print(f"checkpoint trainer={trainer!r} -> model={kind}, num_steps={num_steps}")

    if kind == "routed":
        from nonmarkovian.sample_protein import _build_model_from_ckpt, sample_protein_sequences

        model = _build_model_from_ckpt(cfg, device)
        model.num_timesteps = num_steps
        model.load_state_dict(ckpt["model"])

        def _gen(bs):
            g = torch.Generator(device=device)
            g.manual_seed(args.seed + 7919 * _gen.calls)
            _gen.calls += 1
            return sample_protein_sequences(
                model, num_steps, bs, args.seq_len, device,
                generator=g, history_mode=args.history_mode,
                corruption_mode=args.corruption_mode, release_threshold=args.release_threshold,
            )
    else:
        from nonmarkovian.sample_simple_protein import _build_model_from_ckpt, sample_simple_protein

        model = _build_model_from_ckpt(cfg, device)
        model.num_timesteps = num_steps
        model.load_state_dict(ckpt["model"])

        def _gen(bs):
            g = torch.Generator(device=device)
            g.manual_seed(args.seed + 7919 * _gen.calls)
            _gen.calls += 1
            return sample_simple_protein(model, num_steps, bs, args.seq_len, device, generator=g)

    _gen.calls = 0
    seqs: list[str] = []
    remaining = args.num_seqs
    while remaining > 0:
        bs = min(args.batch, remaining)
        ids = _gen(bs)
        seqs.extend(decode(row.cpu()) for row in ids)
        remaining -= bs
    return seqs


def read_sequences(path: str) -> list[str]:
    out: list[str] = []
    cur = ""
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):  # FASTA header
            if cur:
                out.append(cur)
            cur = ""
        else:
            cur += line
    if cur:
        out.append(cur)
    if not out:  # no FASTA headers -> one sequence per line
        out = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    return out


# --------------------------------------------------------------------------- #
# Fitness: ESM-2 pseudo-perplexity (masked-marginal)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def esm2_pseudo_perplexity(seqs, model_name, device, row_batch=256, fp16=True):
    """Mean masked-marginal pseudo-perplexity per sequence (EvoDiff fitness metric).

    For each position i: mask it, run ESM-2, take -log p(true_aa | rest). ppl = exp(mean_i).

    Batched across **sequences and positions together**: all ``sum_i L_i`` masked variants
    are packed into forwards of up to ``row_batch`` rows (padded to the batch's max length),
    so the big LM runs in a few large batches instead of one tiny batch per sequence.
    ``fp16`` runs the model in half precision on CUDA (~2x faster).
    """
    import esm as esm_lib

    model, alphabet = getattr(esm_lib.pretrained, model_name)()
    model = model.eval().to(device)
    if fp16 and device.type == "cuda":
        model = model.half()
    bc = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx
    pad_idx = alphabet.padding_idx

    cleaned = [_clean(s) for s in seqs]
    # Tokenize each non-empty sequence once -> [1, L+2] (bos/eos).
    toks = {}
    for i, s in enumerate(cleaned):
        if s:
            _, _, t = bc([("p", s)])
            toks[i] = t[0]  # [L_i + 2]

    # Build the flat list of masked variants: (seq_idx, position, true_token).
    variants = [(i, p) for i, s in enumerate(cleaned) if s for p in range(1, len(s) + 1)]
    nll_sum = {i: 0.0 for i in toks}
    nll_cnt = {i: 0 for i in toks}

    for s0 in range(0, len(variants), row_batch):
        chunk = variants[s0 : s0 + row_batch]
        Lmax = max(toks[i].numel() for i, _ in chunk)
        batch = torch.full((len(chunk), Lmax), pad_idx, dtype=torch.long)
        true_tok = torch.empty(len(chunk), dtype=torch.long)
        for r, (i, p) in enumerate(chunk):
            row = toks[i]
            batch[r, : row.numel()] = row
            true_tok[r] = row[p]
            batch[r, p] = mask_idx
        batch = batch.to(device)
        logits = model(batch)["logits"]  # [n, Lmax, V]
        rows = torch.arange(len(chunk))
        pos = torch.tensor([p for _, p in chunk])
        logp = torch.log_softmax(logits[rows, pos].float(), dim=-1)  # [n, V]
        nll = -logp[rows, true_tok.to(device)]  # [n]
        for r, (i, _p) in enumerate(chunk):
            nll_sum[i] += float(nll[r].item())
            nll_cnt[i] += 1

    out = []
    for i in range(len(cleaned)):
        if i in toks and nll_cnt[i] > 0:
            out.append(float(torch.exp(torch.tensor(nll_sum[i] / nll_cnt[i]))))
        else:
            out.append(float("nan"))
    return out


# --------------------------------------------------------------------------- #
# Fitness (alt): ProGen2 autoregressive NLL
# --------------------------------------------------------------------------- #
@torch.no_grad()
def progen2_nll(seqs, model_name, device, fp16=True):
    """Mean autoregressive negative log-likelihood per residue under ProGen2 (nats/residue).

    Unlike ESM-2 pseudo-perplexity (a masked, bidirectional *pseudo*-likelihood), ProGen2 is a
    left-to-right causal LM, so this is a proper chain-rule log-likelihood::

        NLL = mean_i  -log p(aa_i | aa_<i)         (lower = more natural)

    Uses the HuggingFace ``hugohrban/progen2-*`` ports (``trust_remote_code=True``). ProGen2's
    training format brackets the sequence with a BOS token ('1') and EOS token ('2'); the EOS
    prediction is excluded from the per-residue average. Forward direction only (the paper's
    default); scored one sequence per forward to avoid padding-mask subtleties on the 6.4B model.

    Returns the per-sequence NLL list (NaN for empty sequences).
    """
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval().to(device)
    if fp16 and device.type == "cuda":
        model = model.half()
    tokenizer = Tokenizer.from_pretrained(model_name)

    out = []
    for s in (_clean(s) for s in seqs):
        if not s:
            out.append(float("nan"))
            continue
        ids = tokenizer.encode("1" + s + "2").ids  # [BOS] + residues + [EOS]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_ids).logits[0]  # [L, V]
        logp = torch.log_softmax(logits[:-1].float(), dim=-1)  # predict tokens 1..end
        targets = input_ids[0, 1:]  # [r1..rL, EOS]
        nll_tok = -logp[torch.arange(targets.numel(), device=device), targets]
        resid_nll = nll_tok[:-1]  # drop the EOS prediction -> the |s| residues only
        out.append(float(resid_nll.mean().item()))
    return out


# --------------------------------------------------------------------------- #
# Foldability: ESMFold mean pLDDT
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _esmfold_plddt_hf(seqs, device, chunk_size=64):
    """ESMFold via HuggingFace ``EsmForProteinFolding`` (no standalone ``openfold`` needed).

    transformers vendors the openfold utilities (``transformers.models.esm.openfold_utils``),
    so this avoids the fair-esm ``openfold`` build. Returns mean pLDDT (0-100) per sequence,
    averaged over existing atoms then residues.
    """
    from transformers import AutoTokenizer, EsmForProteinFolding

    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval().to(device)
    if device.type == "cuda":
        # Official ESMFold speedup: run the language-model trunk in fp16 (~2x faster).
        model.esm = model.esm.half()
    try:
        model.trunk.set_chunk_size(chunk_size)
    except Exception:
        pass
    out = []
    for seq in seqs:
        if len(seq) < 1:
            out.append(float("nan"))
            continue
        inp = tok([seq], return_tensors="pt", add_special_tokens=False).to(device)
        o = model(**inp)
        plddt = o["plddt"][0].float()  # [L, 37] in 0-100
        mask = o["atom37_atom_exists"][0].float()  # [L, 37]
        per_res = (plddt * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)  # [L]
        val = per_res.mean().item()
        out.append(val * 100.0 if val <= 1.0 else val)
    return out


@torch.no_grad()
def _esmfold_plddt_fairesm(seqs, device, chunk_size=128):
    """ESMFold via fair-esm (requires the standalone ``openfold`` package)."""
    import esm as esm_lib

    model = esm_lib.pretrained.esmfold_v1().eval().to(device)
    try:
        model.set_chunk_size(chunk_size)
    except Exception:
        pass
    out = []
    for seq in seqs:
        if len(seq) < 1:
            out.append(float("nan"))
            continue
        r = model.infer([seq])
        if "mean_plddt" in r:
            out.append(float(r["mean_plddt"][0].item()))
        else:
            out.append(float(r["plddt"][0].mean().item()) * (100.0 if r["plddt"].max() <= 1.0 else 1.0))
    return out


def esmfold_plddt(seqs, device, backend="auto", chunk_size=64):
    """Mean pLDDT (0-100) per sequence. ``backend``: ``hf`` (no openfold) / ``fair-esm`` / ``auto``.

    ``auto`` tries the openfold-free HF implementation first and falls back to fair-esm.
    """
    seqs = [_clean(s) for s in seqs]
    errors = []
    order = {"auto": ("hf", "fair-esm"), "hf": ("hf",), "fair-esm": ("fair-esm",)}[backend]
    for b in order:
        try:
            if b == "hf":
                return _esmfold_plddt_hf(seqs, device, chunk_size)
            return _esmfold_plddt_fairesm(seqs, device, chunk_size)
        except Exception as e:  # noqa: BLE001 - report and try the next backend
            errors.append(f"{b}: {type(e).__name__}: {str(e)[:150]}")
    raise RuntimeError(
        "ESMFold unavailable. Tried [" + " | ".join(errors) + "].\n"
        "Fixes: (a) HF backend needs transformers with a working torch backend "
        "(transformers>=5 requires torch>=2.4; with torch 2.3 use transformers==4.46.*); "
        "(b) fair-esm backend needs the standalone 'openfold' package. "
        "Recommended: a separate eval venv with `pip install 'transformers==4.46.*' accelerate` "
        "(openfold NOT required) and run eval there; keep the training venv untouched."
    )


def _summary(name, values, unit=""):
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return f"{name}: no valid values"
    mean = statistics.fmean(vals)
    med = statistics.median(vals)
    line = f"{name}: mean={mean:.3f}{unit}  median={med:.3f}{unit}  n={len(vals)}"
    return line


def main() -> None:
    p = argparse.ArgumentParser(description="Foldability (ESMFold pLDDT) + fitness (ESM-2 ppl) eval")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str, help="Trained checkpoint to generate from.")
    src.add_argument("--sequences", type=str, help="Score sequences from a FASTA / one-per-line file.")
    p.add_argument("--model", choices=("auto", "routed", "simple"), default="auto",
                   help="Which sampler to use for --checkpoint (auto = from ckpt 'trainer').")
    p.add_argument("--num_seqs", type=int, default=100)
    p.add_argument("--seq_len", type=int, default=200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=0, help="Reverse steps (0 = ckpt num_timesteps).")
    p.add_argument("--history_mode", type=str, default="trajectory", choices=("trajectory", "uniform"))
    p.add_argument("--corruption_mode", type=str, default="independent", choices=("independent", "trajectory"))
    p.add_argument("--release_threshold", type=int, default=6)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    # metrics
    p.add_argument("--esm2_model", type=str, default="esm2_t33_650M_UR50D",
                   help="fair-esm ESM-2 model for pseudo-perplexity (e.g. esm2_t12_35M_UR50D for a fast check).")
    p.add_argument("--ppl_batch", type=int, default=256,
                   help="Masked-variant rows per ESM-2 forward (raise to use more GPU, lower if OOM).")
    p.add_argument("--no_fitness", dest="fitness", action="store_false", default=True,
                   help="Skip ESM-2 pseudo-perplexity.")
    p.add_argument("--progen2", dest="progen2", action="store_true", default=False,
                   help="Also compute ProGen2 autoregressive NLL (true left-to-right likelihood; "
                        "large HF download).")
    p.add_argument("--progen2_model", type=str, default="hugohrban/progen2-xlarge",
                   help="HF ProGen2 port for NLL (e.g. hugohrban/progen2-base for a faster run).")
    p.add_argument("--esmfold", dest="esmfold", action="store_true", default=False,
                   help="Compute ESMFold pLDDT (large model, needs GPU).")
    p.add_argument("--no_esmfold", dest="esmfold", action="store_false")
    p.add_argument(
        "--esmfold_backend", choices=("auto", "hf", "fair-esm"), default="auto",
        help="ESMFold implementation: 'hf' (transformers, no openfold needed), 'fair-esm' "
             "(needs standalone openfold), or 'auto' (try hf then fair-esm).",
    )
    p.add_argument("--out_fasta", type=str, default="", help="Optionally write generated/cleaned seqs here.")
    # reference baseline from the UniRef test split
    p.add_argument("--uniref", type=str, default="auto", help="UniRef dir (for --reference_test).")
    p.add_argument(
        "--reference_test", type=int, default=0,
        help="Also score this many natural sequences from the UniRef TEST split as a "
             "reference baseline (0 = off). Folds/scores them with the same metrics so you "
             "can compare generated pLDDT / perplexity against natural proteins.",
    )
    args = p.parse_args()
    args.rank0 = True

    device = resolve_device_arg(args.device)

    def _score(label: str, raw_seqs: list[str]) -> None:
        cleaned = [_clean(s) for s in raw_seqs]
        nonempty = [s for s in cleaned if s]
        mean_len = statistics.fmean([len(s) for s in nonempty]) if nonempty else 0.0
        print(f"\n[{label}] {len(raw_seqs)} sequences ({len(nonempty)} non-empty canonical); "
              f"mean len={mean_len:.1f}")
        if args.fitness:
            print(f"  [fitness] loading ESM-2 {args.esm2_model} ...")
            ppl = esm2_pseudo_perplexity(cleaned, args.esm2_model, device, row_batch=args.ppl_batch)
            print("  " + _summary("ESM-2 pseudo-perplexity (lower=fitter)", ppl))
        if args.progen2:
            print(f"  [fitness] loading ProGen2 {args.progen2_model} ...")
            pnll = progen2_nll(cleaned, args.progen2_model, device)
            print("  " + _summary("ProGen2 NLL (lower=fitter)", pnll, unit=" nats/res"))
        if args.esmfold:
            print(f"  [foldability] loading ESMFold v1 (backend={args.esmfold_backend}) ...")
            plddt = esmfold_plddt(cleaned, device, backend=args.esmfold_backend)
            print("  " + _summary("ESMFold mean pLDDT (higher=foldable)", plddt))
            valid = [v for v in plddt if v == v]
            good = [v for v in valid if v > 70.0]
            if valid:
                print(f"    fraction pLDDT>70 (foldable): {len(good) / len(valid):.3f}")

    if args.checkpoint:
        seqs = generate_sequences(args, device)
    else:
        seqs = read_sequences(args.sequences)

    if args.out_fasta:
        with open(args.out_fasta, "w") as f:
            for i, s in enumerate(seqs):
                f.write(f">gen_{i}\n{_clean(s)}\n")
        print(f"wrote {args.out_fasta}")

    print("\n=== metrics ===")
    _score("generated", seqs)
    if args.reference_test > 0:
        ref = load_test_sequences(args.uniref, args.reference_test, args.seq_len, args.seed)
        _score("uniref-test (natural reference)", ref)


if __name__ == "__main__":
    main()
