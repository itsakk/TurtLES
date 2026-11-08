"""Zero-parameter baselines.

Both predict the *absolute* next frame, so they are evaluated with
``task.predict_residual=false`` (the evaluation CLI sets this for baselines).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from turtles.models.base import config_from_mapping


@dataclass
class PersistenceConfig:
    """Shape bookkeeping only."""

    space_dim: int = 3
    fun_dim: int = 5
    out_dim: int = 5
    num_steps: int = 1
    num_params: int = 0


@dataclass
class MeanFieldConfig(PersistenceConfig):
    """Shape bookkeeping only."""


class _Baseline(nn.Module):
    def __init__(self, config_cls: type, cfg: Mapping[str, Any] | Any):
        super().__init__()
        self.c = config_from_mapping(config_cls, cfg)
        # A buffer lets `.to(device)` carry a device and keeps the state_dict non-empty.
        self.register_buffer("_anchor", torch.zeros(1), persistent=False)

    @property
    def device(self) -> torch.device:
        """Device the baseline was moved to."""
        return self._anchor.device


class Persistence(_Baseline):
    """``u_{t+1} = u_t``: repeat the last context frame."""

    def __init__(self, cfg: Mapping[str, Any] | PersistenceConfig):
        super().__init__(PersistenceConfig, cfg)

    def forward(
        self,
        context: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor | None = None,
        query_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the last context frame (``query_coords`` must equal ``coords``)."""
        del coords, params, query_coords
        return context[:, -1]


class MeanField(_Baseline):
    """``u_{t+1} = mean_n u_t``: the per-channel spatial mean of the last frame (VRMSE = 1)."""

    def __init__(self, cfg: Mapping[str, Any] | MeanFieldConfig):
        super().__init__(MeanFieldConfig, cfg)

    def forward(
        self,
        context: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor | None = None,
        query_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Broadcast the spatial mean of the last frame to every (query) node."""
        del coords, params
        last = context[:, -1]
        mean = last.mean(dim=1, keepdim=True)
        num_queries = last.shape[1] if query_coords is None else query_coords.shape[1]
        return mean.expand(-1, num_queries, -1).contiguous()
