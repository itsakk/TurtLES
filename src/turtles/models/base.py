"""Shared model plumbing: config parsing and the forward contract."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch

T = TypeVar("T")


def config_from_mapping(config_cls: type[T], cfg: Mapping[str, Any] | T) -> T:
    """Build a dataclass config from a mapping, ignoring keys it does not declare.

    OmegaConf containers are resolved first so interpolations such as
    ``${len:${data.fields}}`` become plain values.
    """
    if isinstance(cfg, config_cls):
        return cfg
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    except ImportError:  # pragma: no cover - omegaconf is a hard dependency
        pass
    if not dataclasses.is_dataclass(config_cls):
        raise TypeError(f"{config_cls} is not a dataclass")
    fields = {f.name for f in dataclasses.fields(config_cls)}
    return config_cls(**{k: v for k, v in dict(cfg).items() if k in fields})  # type: ignore[arg-type]


@runtime_checkable
class Forecaster(Protocol):
    """The single-step evolution operator every model implements."""

    def forward(
        self,
        context: torch.Tensor,
        coords: torch.Tensor,
        params: torch.Tensor | None = None,
        query_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the next frame (or its increment) at ``query_coords`` (default ``coords``)."""
        ...


def flatten_context(context: torch.Tensor) -> torch.Tensor:
    """``(B, T, N, C) -> (B, N, T*C)``: one token per node with time stacked on channels."""
    B, T, N, C = context.shape
    return context.permute(0, 2, 1, 3).reshape(B, N, T * C)
