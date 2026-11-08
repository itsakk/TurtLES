"""Optimizer and learning-rate schedule (linear warmup, then cosine decay)."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR

SCHEDULES: tuple[str, ...] = ("cosine", "constant")


def build_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """AdamW over the trainable parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, betas=tuple(betas), weight_decay=weight_decay)


def lr_multiplier(
    step: int, *, total_steps: int, warmup_steps: int, schedule: str, min_ratio: float
) -> float:
    """Learning-rate multiplier at optimizer step ``step`` (0-indexed)."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    if schedule == "cosine":
        span = max(total_steps - warmup_steps, 1)
        progress = min(max(step - warmup_steps, 0) / span, 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown schedule '{schedule}'; expected one of {SCHEDULES}")


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    schedule: str = "cosine",
    lr_min_ratio: float = 1e-3,
) -> LambdaLR:
    """Per-step ``LambdaLR`` implementing warmup + cosine (or constant) decay."""
    if schedule not in SCHEDULES:
        raise ValueError(f"Unknown schedule '{schedule}'; expected one of {SCHEDULES}")
    return LambdaLR(
        optimizer,
        lambda step: lr_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            schedule=schedule,
            min_ratio=lr_min_ratio,
        ),
    )
