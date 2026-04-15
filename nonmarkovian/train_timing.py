"""Wall-clock timing for training steps (CUDA-synchronized when on GPU)."""

from __future__ import annotations

import time

import torch


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def tic(device: torch.device) -> float:
    sync_device(device)
    return time.perf_counter()


def toc_ms(t0: float, device: torch.device) -> float:
    sync_device(device)
    return (time.perf_counter() - t0) * 1000.0
