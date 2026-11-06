"""Run tracking: distributed-aware running averages and an optional Weights & Biases sink."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class Meter:
    """Running sum / count of one scalar (summed across ranks on :meth:`sync`)."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """Add ``value`` (already averaged over ``n`` items)."""
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        """Mean of everything seen so far."""
        return self.total / self.count if self.count else 0.0


class MeterDict:
    """Named :class:`Meter` collection with one ``update`` entry point.

    Tensor values are converted with ``.item()`` at update time, so callers who want
    to avoid per-step host syncs should buffer tensors and update at flush time.
    """

    def __init__(self) -> None:
        self.meters: dict[str, Meter] = {}

    def update(self, values: Mapping[str, Any], n: int = 1) -> None:
        """Update every meter from a mapping of name to scalar / 0-d tensor."""
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().item()
            self.meters.setdefault(key, Meter()).update(float(value), n)

    def sync(self) -> None:
        """All-reduce totals and counts across DDP ranks."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        keys = sorted(self.meters)
        payload = torch.tensor(
            [[self.meters[k].total, self.meters[k].count] for k in keys],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(payload)
        for i, key in enumerate(keys):
            self.meters[key].total = float(payload[i, 0])
            self.meters[key].count = int(payload[i, 1])

    def averages(self) -> dict[str, float]:
        """Current means, keyed by name."""
        return {k: m.avg for k, m in self.meters.items()}

    def reset(self) -> None:
        """Forget everything."""
        self.meters = {}


class WandbLogger:
    """Thin wrapper around ``wandb`` that is a no-op when disabled or not installed."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str = "TurtLES",
        entity: str | None = None,
        mode: str = "offline",
        directory: str | Path | None = None,
        run_name: str | None = None,
        run_id: str | None = None,
        config: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        self.enabled = bool(enabled)
        self.run = None
        if not self.enabled:
            return
        try:
            import wandb
        except ImportError:
            logger.warning("wandb is not installed; disabling W&B logging")
            self.enabled = False
            return
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
        self.run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            dir=None if directory is None else str(directory),
            name=run_name,
            id=run_id,
            resume="allow",
            config=dict(config) if config is not None else None,
            tags=tags or None,
        )

    @property
    def run_id(self) -> str | None:
        """Identifier to resume the same W&B run later."""
        return None if self.run is None else self.run.id

    def log(self, step: int, scalars: Mapping[str, Any]) -> None:
        """Log a dict of scalars at ``step``."""
        if self.run is None:
            return
        payload = {
            k: (float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v))
            for k, v in scalars.items()
            if isinstance(v, int | float | torch.Tensor)
        }
        if payload:
            self.run.log(payload, step=step)

    def finish(self) -> None:
        """Close the run."""
        if self.run is not None:
            self.run.finish()
            self.run = None
