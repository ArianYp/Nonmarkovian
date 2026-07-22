"""Protein (amino-acid) vocabulary, byte-aligned with the SLM UniRef50 task.

SLM's UniRef pipeline tokenizes with ``evodiff.utils.Tokenizer`` (see
``SLM/dataloader.py:get_tokenizer`` -> ``Tokenizer()``), whose alphabet is the
``sequence_models`` ``MSA_ALPHABET``::

    ACDEFGHIKLMNPQRSTVWY  BZXJOU  - * # @ !
    \\__ 20 canonical __/  \\ amb /  gap stop mask bos pad

This module reproduces that alphabet and the exact integer ids without importing
``evodiff`` / ``sequence_models`` (which live only in the SLM virtualenv), so the
Non-Markovian project can tokenize UniRef sequences identically:

    vocab_size = 31, bos('@')=29, eos('*')=27, mask('#')=28, pad('!')=30

``VOCAB_SIZE`` here is the simplex dimension / model output size used by the
``new_diff`` Bernoulli corruption (over *all* classes, as in SLM
``get_xt_bernoulli`` which one-hots over ``vocab_size``).
"""

from __future__ import annotations

import torch

# sequence_models.constants: ALL_AAS = CAN_AAS + AMB_AAS + OTHER_AAS,
# MSA_ALPHABET = ALL_AAS + GAP + STOP + MASK + START + MSA_PAD
CAN_AAS = "ACDEFGHIKLMNPQRSTVWY"
AMB_AAS = "BZX"
OTHER_AAS = "JOU"
ALL_AAS = CAN_AAS + AMB_AAS + OTHER_AAS  # 26
GAP, STOP, MASK, START, MSA_PAD = "-", "*", "#", "@", "!"

ALPHABET = ALL_AAS + GAP + STOP + MASK + START + MSA_PAD  # "ACDEFGHIKLMNPQRSTVWYBZXJOU-*#@!"
TOKENS = list(ALPHABET)
VOCAB_SIZE = len(TOKENS)  # 31

_A_TO_I = {c: i for i, c in enumerate(TOKENS)}

BOS_IDX = _A_TO_I[START]   # 29
EOS_IDX = _A_TO_I[STOP]    # 27
MASK_IDX = _A_TO_I[MASK]   # 28
PAD_IDX = _A_TO_I[MSA_PAD]  # 30
GAP_IDX = _A_TO_I[GAP]     # 25

IDX_TO_TOKEN = {i: t for i, t in enumerate(TOKENS)}


def token_to_idx(ch: str) -> int:
    return _A_TO_I[ch.upper()]


def encode(seq: str) -> list[int]:
    """Char-level encode a protein string to token ids (``evodiff.Tokenizer.encode``)."""
    return [_A_TO_I[c] for c in seq.upper()]


def decode(ids) -> str:
    """Token ids -> amino-acid string (special tokens included as their chars)."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return "".join(IDX_TO_TOKEN[int(i)] for i in ids)
