"""DNA sequence datasets: synthetic random, FASTA, one-seq-per-line text, or DFM enhancer pickles."""

from __future__ import annotations

import copy
import pickle
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from nonmarkovian.vocab import token_to_idx


def resolve_enhancer_dfm_root(arg: str) -> str:
    """
    Turn CLI ``--enhancer_dfm_root`` into an absolute path.

    - ``auto``: use ``data_dfm`` under the current working directory, or (if missing)
 under the Nonmarkovian repo root (parent of the ``nonmarkovian`` package), if
      ``the_code/General/data/DeepFlyBrain_data.pkl`` exists there.
    - Any other non-empty string: returned as-is (expanded); must contain ``the_code/...``.
    """
    s = arg.strip()
    if not s:
        return ""
    if s.lower() == "auto":
        fly_pkl = ("the_code", "General", "data", "DeepFlyBrain_data.pkl")
        cwd_candidate = Path.cwd() / "data_dfm"
        pkg_root = Path(__file__).resolve().parent.parent
        pkg_candidate = pkg_root / "data_dfm"
        for cand in (cwd_candidate, pkg_candidate):
            if cand.joinpath(*fly_pkl).is_file():
                return str(cand.resolve())
        raise FileNotFoundError(
            "enhancer_dfm_root=auto: expected data_dfm/the_code/General/data/DeepFlyBrain_data.pkl "
            f"under {cwd_candidate} or {pkg_candidate}. Extract Zenodo 10184648 so that path exists."
        )
    return str(Path(s).expanduser().resolve())


class RandomDNADataset(Dataset):
    """Random DNA for sanity checks (no file needed)."""

    def __init__(self, num_samples: int, seq_len: int, num_classes: int = 0, seed: int = 0):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        seq = [self._rng.randint(0, 3) for _ in range(self.seq_len)]
        x = torch.tensor(seq, dtype=torch.long)
        out: dict = {"x0": x}
        if self.num_classes > 0:
            out["label"] = torch.tensor(self._rng.randint(0, self.num_classes - 1), dtype=torch.long)
        return out


class LineDNADataset(Dataset):
    """One DNA string per line (A/C/G/T only). Optional tab-separated label after sequence."""

    def __init__(self, path: str | Path, max_len: int | None = None):
        self.path = Path(path)
        self.max_len = max_len
        self.samples: list[tuple[list[int], int | None]] = []
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                seq_s = parts[0].upper()
                lab: int | None = None
                if len(parts) > 1:
                    lab = int(parts[1])
                chars = [c for c in seq_s if c in "ACGT"]
                if self.max_len is not None:
                    chars = chars[: self.max_len]
                ids = [token_to_idx(c) for c in chars]
                if not ids:
                    continue
                self.samples.append((ids, lab))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ids, lab = self.samples[idx]
        x = torch.tensor(ids, dtype=torch.long)
        out: dict = {"x0": x}
        if lab is not None:
            out["label"] = torch.tensor(lab, dtype=torch.long)
        
        return out


class DFMEnhancerDataset(Dataset):
    """
    Fly-brain or melanoma enhancers from the Dirichlet Flow Matching release
    (Zenodo https://zenodo.org/records/10184648), matching ``EnhancerDataset`` in
    https://github.com/HannesStark/dirichlet-flow-matching/blob/main/utils/dataset.py

    After extracting the Zenodo tarball (e.g. into repo ``data_dfm/``), pass ``root`` as
    the directory that **contains** ``the_code`` — e.g. ``--enhancer_dfm_root data_dfm`` or
    ``auto`` (see ``resolve_enhancer_dfm_root``). Pickles live at
    ``<root>/the_code/General/data/DeepFlyBrain_data.pkl`` and ``DeepMEL2_data.pkl``.
    Sequences are one-hot in the pickle; we convert to ``{0,1,2,3}`` token ids like
    ``LineDNADataset``. Every sample includes a class label.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        mel_enhancer: bool = False,
        max_len: int | None = None,
    ):
        root = Path(root).expanduser().resolve()
        name = "DeepMEL2" if mel_enhancer else "DeepFlyBrain"
        pkl = root / "the_code" / "General" / "data" / f"{name}_data.pkl"
        if not pkl.is_file():
            raise FileNotFoundError(
                f"Expected enhancer pickle at:\n  {pkl}\n"
                "Download Zenodo record 10184648 (see DFM README) and extract so "
                "`the_code/General/data/` exists under the directory you pass as root."
            )
        split = split.strip().lower()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test, got {split!r}")
        with open(pkl, "rb") as f:
            all_data = pickle.load(f)
        data_key = f"{split}_data"
        y_key = f"y_{split}"
        seq_oh = torch.from_numpy(copy.deepcopy(all_data[data_key]))
        y_oh = torch.from_numpy(copy.deepcopy(all_data[y_key]))
        self.seqs = torch.argmax(seq_oh, dim=-1).to(dtype=torch.long)
        self.labels = torch.argmax(y_oh, dim=-1).to(dtype=torch.long)
        self.num_classes: int = int(y_oh.shape[-1])
        self.max_len = max_len

    def __len__(self) -> int:
        return int(self.seqs.shape[0])

    def __getitem__(self, idx: int) -> dict:
        x = self.seqs[idx].clone()
        if self.max_len is not None and x.numel() > self.max_len:
            x = x[: self.max_len]
        return {"x0": x, "label": self.labels[idx].clone()}


def collate_pad(batch: list[dict], pad_idx: int = 0) -> dict:
    """Pad sequences to common length; pad token unused for DNA (use 0=A)."""
    lens = [b["x0"].numel() for b in batch]
    L = max(lens)
    B = len(batch)
    x0 = torch.full((B, L), pad_idx, dtype=torch.long)
    mask_pad = torch.zeros(B, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        li = b["x0"].numel()
        x0[i, :li] = b["x0"]
        mask_pad[i, li:] = True
    out: dict = {"x0": x0, "mask_pad": mask_pad}
    if batch and all("label" in b for b in batch):
        out["label"] = torch.stack([b["label"] for b in batch], dim=0)
    return out


def fasta_to_line_file(fasta_path: str | Path, out_path: str | Path) -> None:
    """Convert multi-line FASTA to one sequence per line (concatenated contig)."""
    fasta_path, out_path = Path(fasta_path), Path(out_path)
    seqs: list[str] = []
    cur: list[str] = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line.upper())
        if cur:
            seqs.append("".join(cur))
    with open(out_path, "w", encoding="utf-8") as o:
        for s in seqs:
            o.write(s + "\n")
