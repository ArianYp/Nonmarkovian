"""UniRef50 protein dataset, byte-aligned with SLM's ``evodiff`` UniRef pipeline.

Mirrors ``SLM/evodiff/data.py`` ``UniRefDataset`` + ``WrappedUniRefDataset``:

- ``UniRefDataset`` reads sequences on demand from ``consensus.fasta`` using the
  byte offsets in ``lengths_and_offsets.npz`` and the index lists in ``splits.json``.
- ``WrappedUniRefDataset`` encodes each sequence with the protein vocab, prepends
  BOS / appends EOS, and pads (or random-crops) to a fixed ``max_len``. It returns
  ``input_ids`` ``[max_len]`` (long) and ``attention_mask`` ``[max_len]`` (1 = real token).

The data layout matches ``data/uniref50/`` in this workspace
(``consensus.fasta``, ``lengths_and_offsets.npz``, ``splits.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from nonmarkovian.vocab_protein import BOS_IDX, EOS_IDX, PAD_IDX, encode


def resolve_uniref_root(arg: str) -> str:
    """Resolve ``--uniref`` to an absolute dir containing ``consensus.fasta``.

    ``auto`` searches ``./data/uniref50`` and ``<workspace>/data/uniref50``.
    """
    s = arg.strip()
    if s and s.lower() != "auto":
        return str(Path(s).expanduser().resolve())
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "data" / "uniref50",
        here.parent.parent.parent / "data" / "uniref50",  # <workspace>/data/uniref50
    ]
    for cand in candidates:
        if (cand / "consensus.fasta").is_file():
            return str(cand.resolve())
    raise FileNotFoundError(
        "uniref=auto: could not find data/uniref50/consensus.fasta under "
        + " or ".join(str(c) for c in candidates)
    )


class UniRefDataset(Dataset):
    """On-demand UniRef sequence reader (SLM ``evodiff.data.UniRefDataset``).

    The folder must contain ``consensus.fasta``, ``lengths_and_offsets.npz`` (with a
    ``seq_offsets`` array) and ``splits.json`` (dict with ``train``/``valid``/``test``).
    Long sequences are random-cropped to ``max_len`` (matching SLM).
    """

    def __init__(self, data_dir: str | Path, split: str, *, max_len: int = 2048):
        self.data_dir = Path(data_dir)
        self.split = split
        self.fasta_path = self.data_dir / "consensus.fasta"
        if not self.fasta_path.is_file():
            raise FileNotFoundError(f"missing {self.fasta_path}")
        with open(self.data_dir / "splits.json", "r") as f:
            splits = json.load(f)
        # SLM/evodiff splits use the key 'valid'; accept 'val' as an alias.
        key = split if split in splits else {"val": "valid", "valid": "val"}.get(split, split)
        if key not in splits:
            raise KeyError(f"split {split!r} (or {key!r}) not in splits.json keys {list(splits)}")
        self.indices = splits[key]
        metadata = np.load(self.data_dir / "lengths_and_offsets.npz")
        self.offsets = metadata["seq_offsets"]
        self.max_len = max_len
        self._fh = None  # lazily opened per worker

    def __len__(self) -> int:
        return len(self.indices)

    def _handle(self):
        if self._fh is None:
            self._fh = open(self.fasta_path, "r")
        return self._fh

    def __getitem__(self, idx: int):
        idx = self.indices[idx]
        offset = self.offsets[idx]
        f = self._handle()
        f.seek(int(offset))
        consensus = f.readline()[:-1]
        if len(consensus) - self.max_len > 0:
            start = np.random.choice(len(consensus) - self.max_len)
            stop = start + self.max_len
        else:
            start = 0
            stop = len(consensus)
        consensus = consensus[start:stop]
        return (consensus,)


def _pad(tokenized: torch.Tensor, max_len: int, value: int) -> torch.Tensor:
    """SLM ``_pad``: right-pad to ``max_len`` (random-crop if longer, keep BOS/EOS)."""
    seq_len = tokenized.shape[0]
    bos_id, eos_id = tokenized[0], tokenized[-1]
    if seq_len > max_len:
        start = np.random.randint(0, seq_len - max_len)
        end = start + max_len
        tokenized = tokenized[start:end]
        tokenized[0] = bos_id
        tokenized[-1] = eos_id
    out = torch.zeros((max_len,), dtype=torch.long) + value
    out[: min(seq_len, max_len)] = tokenized[: min(seq_len, max_len)]
    return out


class WrappedUniRefDataset(Dataset):
    """Tokenize + BOS/EOS + pad, returning ``input_ids`` and ``attention_mask``.

    Byte-aligned with ``SLM/evodiff/data.py:WrappedUniRefDataset`` (encode -> insert
    BOS at front, append EOS, pad to ``max_len`` with PAD; mask = tokens != PAD).
    """

    def __init__(self, dataset: UniRefDataset, max_len: int = 1024):
        self.dataset = dataset
        self.max_length = max_len

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        consensus, *_ = self.dataset[idx]
        tokenized = np.asarray(encode(consensus), dtype=np.int64)
        tokenized = np.insert(tokenized, 0, BOS_IDX)
        tokenized = np.append(tokenized, EOS_IDX)
        tokenized = _pad(torch.from_numpy(tokenized), self.max_length, PAD_IDX)
        attn = (tokenized != PAD_IDX).to(torch.float32)
        return {"x0": tokenized.long(), "attention_mask": attn}


def collate_protein(batch: list[dict]) -> dict:
    """Stack fixed-length protein items into a batch."""
    x0 = torch.stack([b["x0"] for b in batch], dim=0)
    attn = torch.stack([b["attention_mask"] for b in batch], dim=0)
    return {"x0": x0, "attention_mask": attn}
