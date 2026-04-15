"""DNA sequence datasets: synthetic random, FASTA, one-seq-per-line text, or DFM enhancer pickles."""

from __future__ import annotations

import copy
import pickle
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from nonmarkovian.vocab import token_to_idx


def resolve_dfm_enhancer_root(arg: str, *, melanoma: bool = False) -> str:
    """
    Resolve ``--dfm_enhancer`` to an absolute path.

    - ``auto``: ``./data_dfm`` or ``<repo>/data_dfm`` if the expected pickle exists.
    - Any other non-empty string: expanded absolute path (directory that contains ``the_code/``).
    """
    s = arg.strip()
    if not s:
        return ""
    pkl_name = "DeepMEL2_data.pkl" if melanoma else "DeepFlyBrain_data.pkl"
    rel = ("the_code", "General", "data", pkl_name)
    if s.lower() == "auto":
        cwd_candidate = Path.cwd() / "data_dfm"
        pkg_root = Path(__file__).resolve().parent.parent
        pkg_candidate = pkg_root / "data_dfm"
        for cand in (cwd_candidate, pkg_candidate):
            if cand.joinpath(*rel).is_file():
                return str(cand.resolve())
        raise FileNotFoundError(
            f"dfm_enhancer=auto: expected data_dfm/{'/'.join(rel)} under {cwd_candidate} or {pkg_candidate}."
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


def _dfm_pickle_keys_for_split(split: str, all_data: dict) -> tuple[str, str]:
    """Map train|val|test to (seq_key, label_key). Zenodo pickles use valid_data/y_valid, not val_data/y_val."""
    if split == "train":
        candidates = (("train_data", "y_train"),)
    elif split == "val":
        candidates = (("val_data", "y_val"), ("valid_data", "y_valid"))
    elif split == "test":
        candidates = (("test_data", "y_test"),)
    else:
        raise ValueError(f"split must be train|val|test, got {split!r}")
    for dk, yk in candidates:
        if dk in all_data and yk in all_data:
            return dk, yk
    tried = ", ".join(f"{dk}+{yk}" for dk, yk in candidates)
    raise KeyError(
        f"Pickle has no split {split!r} ({tried}); keys: {sorted(all_data.keys())}"
    )


class DFMEnhancerDataset(Dataset):
    """
    Fly-brain or melanoma enhancers from the Dirichlet Flow Matching release
    (Zenodo https://zenodo.org/records/10184648), matching ``EnhancerDataset`` in
    https://github.com/HannesStark/dirichlet-flow-matching/blob/main/utils/dataset.py

    After extracting the Zenodo tarball (e.g. into repo ``data_dfm/``), pass ``root`` as
    the directory that **contains** ``the_code`` (see ``--dfm_enhancer`` in ``train.py``). Pickles live at
    ``<root>/the_code/General/data/DeepFlyBrain_data.pkl`` and ``DeepMEL2_data.pkl``.
    Sequences are one-hot in the pickle; we convert to ``{0,1,2,3}`` token ids like
    ``LineDNADataset``. Every sample includes a class label.
    """

    # One parsed pickle dict per file path (train/val/test datasets share the same file).
    _pickle_cache: dict[str, dict] = {}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        melanoma: bool = False,
        max_len: int | None = None,
    ):
        root = Path(root).expanduser().resolve()
        name = "DeepMEL2" if melanoma else "DeepFlyBrain"
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
        pkl_key = str(pkl.resolve())
        if pkl_key not in DFMEnhancerDataset._pickle_cache:
            with open(pkl, "rb") as f:
                DFMEnhancerDataset._pickle_cache[pkl_key] = pickle.load(f)
        all_data = DFMEnhancerDataset._pickle_cache[pkl_key]
        data_key, y_key = _dfm_pickle_keys_for_split(split, all_data)
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
