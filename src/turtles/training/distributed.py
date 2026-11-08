"""Process-group initialization that works under torchrun, srun and submitit alike."""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistInfo:
    """Rank layout of the current process."""

    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        """True on the process that logs and writes checkpoints."""
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        """True when more than one process participates."""
        return self.world_size > 1


def _env_int(*names: str, default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return default


def init_distributed(*, use_cuda: bool = True) -> tuple[DistInfo, torch.device]:
    """Read the rank layout from the environment, pick the device, init NCCL if needed.

    Honors ``WORLD_SIZE/RANK/LOCAL_RANK`` (torchrun) with ``SLURM_NTASKS/
    SLURM_PROCID/SLURM_LOCALID`` fallbacks (srun, submitit).
    """
    world_size = _env_int("WORLD_SIZE", "SLURM_NTASKS", default=1)
    rank = _env_int("RANK", "SLURM_PROCID", default=0) if world_size > 1 else 0
    local_rank = _env_int("LOCAL_RANK", "SLURM_LOCALID", default=0) if world_size > 1 else 0
    info = DistInfo(rank=rank, local_rank=local_rank, world_size=world_size)

    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", local_rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if info.distributed and not dist.is_initialized():
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        logger.info(
            "Distributed: rank %d/%d (local %d) on %s",
            rank,
            world_size,
            local_rank,
            socket.gethostname(),
        )
    return info, device


def cleanup_distributed() -> None:
    """Destroy the process group if one was created."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronize all ranks (no-op when not distributed)."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
