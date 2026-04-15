#!/usr/bin/env python3
"""
Convert genomic intervals (BED) + *Drosophila melanogaster* dm6 FASTA into one-line DNA
files for nonmarkovian.train LineDNADataset (unconditional enhancer / cCRE training).

Source paper: Janssens, Aibar, Taskiran et al., "Decoding gene regulation in the fly brain"
(Nature 2022, doi:10.1038/s41586-021-04262-z). Raw snATAC-seq: GEO GSE163697 / SRP298930.
Regulatory region coordinates are not shipped as plain FASTA in GEO; use author-provided
BED / supplementary region tables or exported peaks, then extract sequence with this script.

Requirements:
  - Reference genome FASTA for the same assembly as the BED (typically UCSC dm6).
  - BED with at least chrom, start, end (0-based half-open, standard BED).

Example (GEO narrowPeak.gz, e.g. GSE163697 differential peaks):
  python scripts/bed_to_training_lines.py \\
    --bed peaks1.narrowPeak.gz peaks2.narrowPeak.gz \\
    --genome dm6.fa \\
    --out data/flybrain_enhancers_lines.txt \\
    --max-len 256
"""

from __future__ import annotations

import argparse
import gzip
from collections.abc import Iterator
from pathlib import Path


def resolve_chrom_name(chrom: str, genome: dict[str, str]) -> str | None:
    if chrom in genome:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in genome:
        return chrom[3:]
    alt = f"chr{chrom}" if not chrom.startswith("chr") else chrom
    if alt in genome:
        return alt
    return None


def _open_text(path: Path):
    if path.suffix.lower() == ".gz" and path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_fasta(path: Path) -> dict[str, str]:
    """Load a (possibly multi-GB) FASTA into chrom -> sequence. Uppercases ACGTN."""
    chroms: dict[str, list[str]] = {}
    cur: str | None = None
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                hdr = line[1:].split()[0]
                cur = hdr
                chroms[cur] = []
            elif cur is not None:
                chroms[cur].append(line.upper())
    return {k: "".join(v) for k, v in chroms.items()}


def iter_bed_intervals(path: Path) -> Iterator[tuple[str, int, int]]:
    """BED or narrowPeak (first 3 columns = chrom, start, end); supports .gz."""
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            chrom, start_s, end_s = parts[0], parts[1], parts[2]
            yield chrom, int(start_s), int(end_s)


def iter_bed_intervals_many(paths: list[Path]) -> Iterator[tuple[str, int, int]]:
    for p in paths:
        yield from iter_bed_intervals(p)


def center_crop(seq: str, max_len: int) -> str:
    if len(seq) <= max_len:
        return seq
    excess = len(seq) - max_len
    start = excess // 2
    return seq[start : start + max_len]


def main() -> None:
    p = argparse.ArgumentParser(description="BED + genome FASTA -> one DNA sequence per line (ACGT)")
    p.add_argument(
        "--bed",
        type=Path,
        nargs="+",
        required=True,
        help="BED or narrowPeak (first 3 columns); use one or more files (.gz ok)",
    )
    p.add_argument("--genome", type=Path, required=True, help="dm6.fa[.gz] (same assembly as intervals)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-len", type=int, default=0, help="If >0, center-crop longer sequences")
    p.add_argument("--min-len", type=int, default=1, help="Skip intervals shorter than this")
    p.add_argument(
        "--skip-n-fraction",
        type=float,
        default=1.0,
        help="Skip sequence if fraction of N bases exceeds this (default 1.0 = never skip)",
    )
    args = p.parse_args()

    print("loading genome FASTA (this may take a minute)...")
    genome = parse_fasta(args.genome)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    n_skip_short = 0
    n_skip_n = 0
    n_skip_missing = 0

    with open(args.out, "w", encoding="utf-8") as out:
        for chrom, start, end in iter_bed_intervals_many(args.bed):
            if end <= start:
                continue
            key = resolve_chrom_name(chrom, genome)
            if key is None:
                n_skip_missing += 1
                continue
            seq = genome[key][start:end]
            if len(seq) < args.min_len:
                n_skip_short += 1
                continue
            n_count = seq.count("N") + seq.count("n")
            if len(seq) > 0 and n_count / len(seq) > args.skip_n_fraction:
                n_skip_n += 1
                continue
            # Keep only ACGT for consistency with LineDNADataset
            seq = "".join(c for c in seq.upper() if c in "ACGT")
            if not seq:
                continue
            if args.max_len > 0:
                seq = center_crop(seq, args.max_len)
            out.write(seq + "\n")
            n_out += 1

    print(f"wrote {n_out} sequences to {args.out}")
    if n_skip_missing:
        print(f"skipped {n_skip_missing} intervals (chrom not in FASTA — check chr naming vs dm6)")
    if n_skip_short:
        print(f"skipped {n_skip_short} intervals (shorter than --min-len)")
    if n_skip_n:
        print(f"skipped {n_skip_n} intervals (too many N)")


if __name__ == "__main__":
    main()
