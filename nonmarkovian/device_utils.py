"""Pick CPU vs CUDA without noisy warnings when the NVIDIA driver is too old for this PyTorch build."""

from __future__ import annotations

import warnings

import torch


def cuda_is_usable() -> bool:
    """Return True only if a CUDA tensor can be allocated (driver matches PyTorch CUDA)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            if not torch.cuda.is_available():
                return False
            torch.zeros(1, device="cuda")
            return True
        except Exception:
            return False


def resolve_device_arg(device_str: str) -> torch.device:
    """
    - auto: use CUDA if usable, else CPU.
    - cuda / cuda:N: use that device if usable, else CPU with a one-line message.
    - cpu: always CPU.
    """
    if device_str == "auto":
        return torch.device("cuda" if cuda_is_usable() else "cpu")
    d = torch.device(device_str)
    if d.type == "cuda" and not cuda_is_usable():
        print("CUDA requested but not usable (driver too old for this PyTorch build, or no GPU); using CPU.")
        return torch.device("cpu")
    return d
