"""Training loop, optimizer schedule, checkpointing and distributed setup."""

from __future__ import annotations

from turtles.training.checkpoint import (
    load_checkpoint,
    load_model_weights,
    resolve_checkpoint,
    save_checkpoint,
    strip_prefixes,
)
from turtles.training.distributed import DistInfo, cleanup_distributed, init_distributed
from turtles.training.schedule import build_lr_scheduler, build_optimizer
from turtles.training.trainer import Trainer, TrainerConfig

__all__ = [
    "DistInfo",
    "Trainer",
    "TrainerConfig",
    "build_lr_scheduler",
    "build_optimizer",
    "cleanup_distributed",
    "init_distributed",
    "load_checkpoint",
    "load_model_weights",
    "resolve_checkpoint",
    "save_checkpoint",
    "strip_prefixes",
]
