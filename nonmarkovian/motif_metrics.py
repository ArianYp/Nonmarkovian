"""Do generated enhancers contain the right transcription-factor motifs?

A model-agnostic sequence metric for comparing methods (ours vs baselines) that does not depend on
a learned scorer. Every set of sequences is scanned for a library of TF motifs and compared against
the real data's motif profile.

Why this metric
---------------
The FBD classifier can only be trusted distributionally: its class head puts real held-out
fly-brain sequences below chance under their own labels, and per-sequence embedding proximity ranks
randomised real sequences above real ones. Motif hits have no such problem -- a match either clears
the log-odds threshold or it does not -- so the measurement is interpretable at any sample size and
its positive control (shuffling must destroy motifs) is trivial to verify.

Scanning
--------
Each motif is a position weight matrix. It is converted to a log-odds matrix against a background
(the observed base composition of the *real* sequences, so an AT-rich genome does not inflate
AT-rich motifs), then applied to one-hot sequences as a ``conv1d`` on **both strands**. A position
counts as a hit when its score clears the motif's own detection threshold -- HOMER ``.motif`` files
carry that threshold on the header line, which avoids inventing a cutoff.

Reported per sequence set
-------------------------
* ``hit_rate``      -- fraction of sequences containing the motif at least once
* ``hits_per_seq``  -- mean number of hits per sequence
* versus real: ``log2 fold-change`` per motif, the Pearson/Spearman correlation between the set's
  motif-frequency profile and the real one, and ``mean |log2 FC|`` as a single summary number.

A **dinucleotide-shuffled** copy of the real sequences is scored automatically as the floor: it
preserves base and dinucleotide composition while destroying motif structure, so it shows how much
of any method's motif content is explained by composition alone. A method that cannot beat the
shuffle is not encoding motifs, whatever its raw hit counts look like.

Usage::

    python -m nonmarkovian.motif_metrics \
        --motif_dir data_motifs/the_code/Fly/data/homer/M0_vs_M10/knownResults \
        --real_split test \
        --sets nonmarkov=samples_nm.txt markov=samples_markov.txt simple=samples_simple.txt \
        --out_dir logs/motifs

Sequence files are plain text, one ACGT sequence per line (what ``sample.py --out`` writes); FASTA
is accepted too.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from nonmarkovian.device_utils import resolve_device_arg
from nonmarkovian.vocab import TOKENS

NUC = ("A", "C", "G", "T")
_COMPLEMENT = {0: 3, 1: 2, 2: 1, 3: 0}


def parse_homer_motif(path: Path) -> dict | None:
    """Parse a HOMER ``.motif`` file: header ``>consensus<TAB>name<TAB>threshold`` then a PWM."""
    lines = [ln.rstrip("\n") for ln in path.read_text().splitlines() if ln.strip()]
    if not lines or not lines[0].startswith(">"):
        return None
    head = lines[0][1:].split("\t")
    consensus = head[0].strip() if head else path.stem
    name = head[1].strip() if len(head) > 1 else path.stem
    try:
        threshold = float(head[2])
    except (IndexError, ValueError):
        threshold = None
    rows = []
    for ln in lines[1:]:
        parts = ln.replace(",", " ").split()
        if len(parts) < 4:
            continue
        try:
            rows.append([float(x) for x in parts[:4]])
        except ValueError:
            continue
    if not rows:
        return None
    return {
        "name": name,
        "consensus": consensus,
        "threshold": threshold,
        "pwm": torch.tensor(rows, dtype=torch.float64),   # [w, 4] probabilities
        "source": str(path),
    }


def load_motifs(motif_dir: Path, limit: int = 0) -> list[dict]:
    files = sorted(
        motif_dir.rglob("*.motif"),
        key=lambda p: (int(m.group(1)) if (m := re.search(r"(\d+)", p.stem)) else 0, p.stem),
    )
    motifs = []
    seen_names = set()
    for f in files:
        m = parse_homer_motif(f)
        if m is None:
            continue
        if m["name"] in seen_names:      # HOMER repeats motifs across comparisons
            continue
        seen_names.add(m["name"])
        motifs.append(m)
        if limit and len(motifs) >= limit:
            break
    return motifs


def read_sequences(path: Path, max_len: int = 0) -> torch.Tensor:
    """Plain-text (one sequence per line) or FASTA -> ``[N, L]`` uint8 ids."""
    idx = {c: i for i, c in enumerate(NUC)}
    lines = path.read_text().splitlines()
    # Without '>' headers there is no record structure to join across, so every line is its own
    # sequence -- which is what sample.py --out and --dump_sequences write. Joining unconditionally
    # would fold the whole file into one record and max_len would then keep only the first sequence.
    is_fasta = any(ln.startswith(">") for ln in lines)
    seqs, cur = [], []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
            continue
        if not set(ln.upper()) <= set("ACGTN"):  # not sequence content
            continue
        if is_fasta:
            cur.append(ln.upper())
        else:
            seqs.append(ln.upper())
    if cur:
        seqs.append("".join(cur))
    if not seqs:
        raise SystemExit(f"No sequences found in {path}")
    L = min(len(s) for s in seqs)
    if max_len:
        L = min(L, max_len)
    out = torch.zeros((len(seqs), L), dtype=torch.uint8)
    for i, s in enumerate(seqs):
        out[i] = torch.tensor([idx.get(c, 0) for c in s[:L]], dtype=torch.uint8)
    return out


def dinucleotide_shuffle(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Altschul-Erikson dinucleotide shuffle per sequence (preserves dinucleotide counts).

    Implemented as a random Eulerian walk on the 4-node dinucleotide graph: shuffle the outgoing
    edge list of every base, then traverse. Keeps composition *and* dinucleotide frequencies, so
    what it destroys is motif structure specifically.
    """
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        s = x[i].tolist()
        edges: dict[int, list[int]] = {c: [] for c in range(4)}
        for a, b in zip(s[:-1], s[1:]):
            edges[a].append(b)
        for c in range(4):
            e = edges[c]
            if len(e) > 1:
                perm = torch.randperm(len(e), generator=gen).tolist()
                edges[c] = [e[p] for p in perm]
        pos = {c: 0 for c in range(4)}
        cur = s[0]
        walk = [cur]
        for _ in range(len(s) - 1):
            e = edges[cur]
            if pos[cur] >= len(e):          # dead end: restart from any base with edges left
                cand = [c for c in range(4) if pos[c] < len(edges[c])]
                if not cand:
                    walk.extend(s[len(walk):])
                    break
                cur = cand[0]
                e = edges[cur]
            nxt = e[pos[cur]]
            pos[cur] += 1
            walk.append(nxt)
            cur = nxt
        out[i] = torch.tensor(walk[: x.shape[1]], dtype=torch.uint8)
    return out


class MotifScanner:
    """Log-odds PWM scanning of one-hot sequences on both strands, as a single conv1d per motif."""

    def __init__(self, motifs: list[dict], background: torch.Tensor, device: torch.device,
                 pseudocount: float = 1e-3, threshold_scale: float = 1.0) -> None:
        self.motifs = motifs
        self.device = device
        self.names = [m["name"] for m in motifs]
        bg = background.to(torch.float64).clamp(min=1e-6)
        bg = bg / bg.sum()
        self.background = bg
        self.kernels, self.thresholds, self.widths = [], [], []
        for m in motifs:
            p = m["pwm"].clamp(min=pseudocount)
            p = p / p.sum(dim=-1, keepdim=True)
            lo = torch.log2(p / bg.unsqueeze(0))                    # [w, 4] log-odds (base 2)
            fwd = lo.T.to(torch.float32)                            # [4, w]
            rev = torch.flip(lo[:, [3, 2, 1, 0]].T, dims=[1]).to(torch.float32)
            self.kernels.append(torch.stack([fwd, rev]).to(device))  # [2, 4, w]
            thr = m["threshold"]
            if thr is None:
                # HOMER thresholds are natural-log odds vs a uniform background; without one, fall
                # back to 80% of the motif's maximum attainable score (a common default).
                thr = 0.8 * float(lo.max(dim=-1).values.sum())
            else:
                thr = float(thr) / math.log(2.0)                     # HOMER stores ln, we score log2
            self.thresholds.append(thr * float(threshold_scale))
            self.widths.append(p.shape[0])

    @torch.no_grad()
    def hits(self, seqs: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """``[N, L]`` ids -> ``[N, n_motifs]`` hit counts (both strands, threshold per motif)."""
        counts = torch.zeros((seqs.shape[0], len(self.motifs)), dtype=torch.int32)
        for i in range(0, seqs.shape[0], chunk):
            block = seqs[i : i + chunk].long().to(self.device)
            oh = F.one_hot(block, num_classes=4).permute(0, 2, 1).float()   # [B, 4, L]
            for j, (k, thr) in enumerate(zip(self.kernels, self.thresholds)):
                sc = F.conv1d(oh, k)                                        # [B, 2, L-w+1]
                # Best strand per offset, then threshold: a site occupied on both strands (every
                # palindromic motif) is one binding site, not two.
                best = sc.max(dim=1).values                                  # [B, L-w+1]
                counts[i : i + chunk, j] = (best >= thr).sum(dim=1).cpu().to(torch.int32)
        return counts


def summarise(counts: torch.Tensor, names: list[str]) -> dict:
    n = counts.shape[0]
    present = (counts > 0).to(torch.float64)
    return {
        "n_sequences": int(n),
        "hit_rate": present.mean(dim=0).tolist(),
        "hits_per_seq": counts.to(torch.float64).mean(dim=0).tolist(),
        "total_hits_per_seq": float(counts.sum(dim=1).to(torch.float64).mean()),
        "mean_motifs_present_per_seq": float(present.sum(dim=1).mean()),
        "names": names,
    }


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(a.norm() * b.norm())
    return float((a @ b) / d) if d > 0 else float("nan")


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().to(torch.float64)
    rb = b.argsort().argsort().to(torch.float64)
    return _pearson(ra, rb)


def compare(real: dict, other: dict, min_real_hits: float = 0.01) -> dict:
    r = torch.tensor(real["hits_per_seq"], dtype=torch.float64)
    o = torch.tensor(other["hits_per_seq"], dtype=torch.float64)
    keep = r >= float(min_real_hits)          # motifs the real data actually uses
    eps = 1e-6
    lfc = torch.log2((o[keep] + eps) / (r[keep] + eps))
    rr = torch.tensor(real["hit_rate"], dtype=torch.float64)[keep]
    oo = torch.tensor(other["hit_rate"], dtype=torch.float64)[keep]
    return {
        "n_motifs_compared": int(keep.sum()),
        "pearson_hits_per_seq": _pearson(r[keep], o[keep]),
        "spearman_hits_per_seq": _spearman(r[keep], o[keep]),
        "pearson_hit_rate": _pearson(rr, oo),
        "mean_abs_log2_fc": float(lfc.abs().mean()),
        "median_log2_fc": float(lfc.median()),
        "total_hits_ratio": float(o[keep].sum() / r[keep].sum().clamp(min=eps)),
        "n_motifs_depleted_2x": int((lfc < -1).sum()),
        "n_motifs_enriched_2x": int((lfc > 1).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare generated sequence sets against real data by TF-motif content.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--motif_dir", type=str, required=True, help="Directory of HOMER .motif files.")
    p.add_argument("--max_motifs", type=int, default=0, help="0 = all.")
    p.add_argument(
        "--sets", type=str, nargs="+", required=True,
        help="label=path.txt entries; one ACGT sequence per line (or FASTA).",
    )
    p.add_argument("--real", type=str, default="", help="Real sequences file; omit to use the data split.")
    p.add_argument("--real_split", type=str, default="test", choices=("val", "test"))
    p.add_argument("--dfm_enhancer", type=str, default="")
    p.add_argument("--dfm_melanoma", dest="dfm_melanoma", action="store_true", default=None)
    p.add_argument("--no_dfm_melanoma", dest="dfm_melanoma", action="store_false")
    p.add_argument("--n_real", type=int, default=4096)
    p.add_argument("--max_len", type=int, default=500)
    p.add_argument(
        "--no_shuffle_control", action="store_true",
        help="Skip the dinucleotide-shuffled floor (it is the control that makes hit counts "
        "interpretable, so only skip it to save time).",
    )
    p.add_argument(
        "--threshold_scale", type=float, default=1.0,
        help="Multiply every motif's detection threshold (>1 = stricter).",
    )
    p.add_argument("--min_real_hits", type=float, default=0.01,
                   help="Ignore motifs with fewer than this many hits/sequence in the real data.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="logs/motifs")
    p.add_argument("--tag", type=str, default="")
    cli = p.parse_args()

    device = resolve_device_arg(cli.device)
    motifs = load_motifs(Path(cli.motif_dir), limit=int(cli.max_motifs))
    if not motifs:
        raise SystemExit(f"No parsable .motif files under {cli.motif_dir}")

    # --- real reference ---
    if cli.real:
        real_seqs = read_sequences(Path(cli.real), max_len=int(cli.max_len))
    else:
        from nonmarkovian.eval_checkpoint import _build_loader

        cfg = {
            "dfm_enhancer": cli.dfm_enhancer or "auto",
            "dfm_melanoma": bool(cli.dfm_melanoma),
            "max_len": int(cli.max_len),
        }
        loader = _build_loader(
            cfg, cli.real_split, batch_size=512,
            dfm_root_override=cli.dfm_enhancer, melanoma_override=cli.dfm_melanoma,
        )
        chunks, got = [], 0
        for b in loader:
            take = min(b["x0"].shape[0], int(cli.n_real) - got)
            chunks.append(b["x0"][:take].to(torch.uint8))
            got += take
            if got >= int(cli.n_real):
                break
        real_seqs = torch.cat(chunks)
    real_seqs = real_seqs[:, : int(cli.max_len)]

    # Background = real base composition, so composition alone cannot inflate a motif.
    bg = torch.bincount(real_seqs.reshape(-1).long(), minlength=4).to(torch.float64)
    scanner = MotifScanner(motifs, bg, device, threshold_scale=float(cli.threshold_scale))
    print(f"[motif] {len(motifs)} motifs; real background A/C/G/T = "
          f"{(bg / bg.sum()).mul(100).round().to(torch.int).tolist()}%", flush=True)

    sets: dict[str, torch.Tensor] = {"real": real_seqs}
    if not cli.no_shuffle_control:
        gen = torch.Generator().manual_seed(int(cli.seed))
        sets["real_dinuc_shuffled"] = dinucleotide_shuffle(real_seqs, gen)
    for spec in cli.sets:
        if "=" not in spec:
            raise SystemExit(f"--sets entries must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        sets[label] = read_sequences(Path(path), max_len=int(cli.max_len))

    summaries: dict[str, dict] = {}
    for label, seqs in sets.items():
        counts = scanner.hits(seqs)
        summaries[label] = summarise(counts, scanner.names)
        print(f"[motif] {label:<24} n={seqs.shape[0]:<6} "
              f"total hits/seq={summaries[label]['total_hits_per_seq']:.2f}  "
              f"motifs present/seq={summaries[label]['mean_motifs_present_per_seq']:.2f}", flush=True)

    comparisons = {
        label: compare(summaries["real"], s, min_real_hits=float(cli.min_real_hits))
        for label, s in summaries.items()
        if label != "real"
    }

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (cli.tag + "_") if cli.tag else ""
    with open(out_dir / f"{tag}motif_metrics.json", "w") as f:
        json.dump({"summaries": summaries, "comparisons": comparisons,
                   "config": {"motif_dir": cli.motif_dir, "n_motifs": len(motifs),
                              "threshold_scale": cli.threshold_scale,
                              "real_split": cli.real_split}}, f, indent=2)
    with open(out_dir / f"{tag}motif_per_motif.csv", "w") as f:
        f.write("motif," + ",".join(f"{k}_hits_per_seq,{k}_hit_rate" for k in summaries) + "\n")
        for i, name in enumerate(scanner.names):
            row = [name.replace(",", ";")]
            for k in summaries:
                row += [f"{summaries[k]['hits_per_seq'][i]:.6f}", f"{summaries[k]['hit_rate'][i]:.6f}"]
            f.write(",".join(row) + "\n")

    print("\n[motif] === MOTIF PROFILE vs REAL ===")
    print(f"  {'set':<24}{'pearson':>9}{'spearman':>10}{'mean|log2FC|':>14}"
          f"{'hits ratio':>12}{'dep2x':>7}{'enr2x':>7}")
    for label, c in comparisons.items():
        print(f"  {label:<24}{c['pearson_hits_per_seq']:>9.3f}{c['spearman_hits_per_seq']:>10.3f}"
              f"{c['mean_abs_log2_fc']:>14.3f}{c['total_hits_ratio']:>12.3f}"
              f"{c['n_motifs_depleted_2x']:>7}{c['n_motifs_enriched_2x']:>7}")
    if "real_dinuc_shuffled" in comparisons:
        sh = comparisons["real_dinuc_shuffled"]
        print(f"\n  floor: dinucleotide-shuffled real data scores pearson="
              f"{sh['pearson_hits_per_seq']:.3f}, mean|log2FC|={sh['mean_abs_log2_fc']:.3f}, "
              f"hits ratio={sh['total_hits_ratio']:.3f}")
        print("  A method must beat this to be encoding motifs rather than base composition.")
    print(f"\n[motif] wrote {out_dir / f'{tag}motif_metrics.json'}")


if __name__ == "__main__":
    main()
