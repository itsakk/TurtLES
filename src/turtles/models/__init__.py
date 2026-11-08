"""Model registry.

Every model follows one forward contract (see :mod:`turtles.models.base`)::

    forward(context, coords, params=None, query_coords=None) -> prediction
        context:      (B, T_in, N, C)  normalized fields of the context frames
        coords:       (B, N, 3)        mesh node coordinates
        params:       (B, P) or None   normalized PDE parameters
        query_coords: (B, Q, 3) or None  where to predict (default: coords)
        returns:      (B, Q, C)        next frame (or its increment)

Models are imported lazily by name so that optional dependencies (flashlib,
neuraloperator, triton) are only needed for the model actually requested.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import torch

from turtles.models.base import config_from_mapping

# name -> (module, model class, config class)
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "calm_pde": ("turtles.models.calm_pde", "Model", "Config"),
    "upt": ("turtles.models.upt", "Model", "Config"),
    "transolver": ("turtles.models.transolver", "Model", "Config"),
    "gino": ("turtles.models.gino", "Model", "Config"),
    "persistence": ("turtles.models.baselines", "Persistence", "PersistenceConfig"),
    "mean_field": ("turtles.models.baselines", "MeanField", "MeanFieldConfig"),
}

MODEL_NAMES: tuple[str, ...] = tuple(_REGISTRY)

__all__ = ["MODEL_NAMES", "build_model", "count_parameters", "resolve"]


def resolve(name: str) -> tuple[type[torch.nn.Module], type]:
    """Return ``(model_class, config_class)`` for a registered model name."""
    try:
        module_name, model_attr, config_attr = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(MODEL_NAMES)}") from exc
    module = importlib.import_module(module_name)
    return getattr(module, model_attr), getattr(module, config_attr)


def build_model(cfg: Mapping[str, Any]) -> torch.nn.Module:
    """Instantiate the model described by a ``model`` config section (needs ``name``)."""
    model_cls, config_cls = resolve(str(cfg["name"]))
    return model_cls(config_from_mapping(config_cls, cfg))


def count_parameters(model: torch.nn.Module, *, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
