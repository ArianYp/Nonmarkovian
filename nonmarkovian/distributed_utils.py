"""Single-node multi-GPU helpers (torchrun / DistributedDataParallel)."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    if v == "":
        return default
    return int(v)


def world_size_from_env() -> int:
    return _env_int("WORLD_SIZE", 1)


def local_rank_from_env() -> int:
    return _env_int("LOCAL_RANK", 0)


def setup_process_group() -> tuple[bool, int, int, int]:
    """
    If WORLD_SIZE > 1, init NCCL (CUDA) or gloo (CPU) and return
    (True, rank, world_size, local_rank). Otherwise (False, 0, 1, 0).
    Call once per process before building models.
    """
    ws = world_size_from_env()
    if ws <= 1:
        return False, 0, 1, 0
    rank = _env_int("RANK", 0)
    local_rank = local_rank_from_env()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        dist.init_process_group(backend="gloo")
    return True, rank, ws, local_rank


def cleanup_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """Return inner module if wrapped with DistributedDataParallel."""
    return module.module if hasattr(module, "module") else module
