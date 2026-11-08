"""Field and parameter normalization.

Fields span very different scales (``p, rho ≈ 1 ± 0.02`` versus ``w ≈ 0 ± 0.03``), so
models train on per-field normalized values taken from the ``stats.yaml`` shipped
with the dataset (statistics of the train split). Evaluation always converts
predictions back to physical units before scoring.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from turtles.data.cases import CaseParams

STATS_FILENAME = "stats.yaml"
STRATEGIES: tuple[str, ...] = ("none", "zscore", "zscore_delta", "rms")

ArrayLike = np.ndarray | torch.Tensor


def load_stats(path: str | Path) -> dict[str, dict[str, float]]:
    """Load ``stats.yaml`` (``{statistic: {field: value}}``)."""
    with open(path) as handle:
        stats = yaml.safe_load(handle)
    if not isinstance(stats, Mapping) or "mean" not in stats:
        raise ValueError(f"{path} does not look like a TurtLES stats file")
    return {k: dict(v) for k, v in stats.items()}


class FieldNormalizer:
    """Affine per-field normalization ``(x - shift) / scale`` on the last axis.

    Strategies:
        ``none``: identity.
        ``zscore``: ``(x - mean) / std``; balances the *state* scales.
        ``zscore_delta``: ``(x - mean) / std_delta``; balances the *increment* scales,
        the natural choice when the model predicts ``u_{t+1} - u_t``.
        ``rms``: ``x / rms``; a pure rescaling that keeps zero special.
    """

    def __init__(self, shift: np.ndarray | None, scale: np.ndarray | None, strategy: str = "none"):
        self.strategy = strategy
        self.shift = None if shift is None else np.asarray(shift, dtype=np.float32)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float32)
        if (self.shift is None) != (self.scale is None):
            raise ValueError("shift and scale must both be set or both be None")
        if self.scale is not None and np.any(self.scale <= 0):
            raise ValueError(f"normalization scales must be positive, got {self.scale}")

    @classmethod
    def identity(cls) -> FieldNormalizer:
        """A normalizer that leaves values untouched."""
        return cls(None, None, "none")

    @classmethod
    def from_stats(
        cls, stats: Mapping[str, Mapping[str, float]], fields: Sequence[str], strategy: str | None
    ) -> FieldNormalizer:
        """Build a normalizer for ``fields`` from loaded statistics."""
        strategy = (strategy or "none").lower()
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown normalization '{strategy}'; expected one of {STRATEGIES}")
        if strategy == "none":
            return cls.identity()
        if strategy == "zscore":
            shift, scale = stats["mean"], stats["std"]
        elif strategy == "zscore_delta":
            shift, scale = stats["mean"], stats["std_delta"]
        else:  # rms
            shift, scale = dict.fromkeys(fields, 0.0), stats["rms"]
        return cls(
            np.array([shift[f] for f in fields], dtype=np.float32),
            np.array([scale[f] for f in fields], dtype=np.float32),
            strategy,
        )

    @classmethod
    def from_file(
        cls, path: str | Path, fields: Sequence[str], strategy: str | None
    ) -> FieldNormalizer:
        """Build a normalizer from a ``stats.yaml`` file (identity when ``strategy`` is none)."""
        if (strategy or "none").lower() == "none":
            return cls.identity()
        return cls.from_stats(load_stats(path), fields, strategy)

    @property
    def enabled(self) -> bool:
        """False for the identity normalizer."""
        return self.scale is not None

    def _coefficients(self, x: ArrayLike) -> tuple[Any, Any]:
        if isinstance(x, torch.Tensor):
            shift = torch.as_tensor(self.shift, device=x.device, dtype=x.dtype)
            scale = torch.as_tensor(self.scale, device=x.device, dtype=x.dtype)
            return shift, scale
        return self.shift, self.scale

    def normalize(self, x: ArrayLike) -> ArrayLike:
        """Physical units -> normalized values (last axis = fields)."""
        if self.scale is None:
            return x
        shift, scale = self._coefficients(x)
        return (x - shift) / scale

    def unnormalize(self, x: ArrayLike) -> ArrayLike:
        """Normalized values -> physical units (exact inverse of :meth:`normalize`)."""
        if self.scale is None:
            return x
        shift, scale = self._coefficients(x)
        return x * scale + shift

    def state_dict(self) -> dict[str, Any]:
        """Serializable description (stored in checkpoints)."""
        return {
            "strategy": self.strategy,
            "shift": None if self.shift is None else self.shift.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> FieldNormalizer:
        """Inverse of :meth:`state_dict`."""
        return cls(state.get("shift"), state.get("scale"), str(state.get("strategy", "none")))


# Sampling ranges of the dataset (paper, Section 3): used to map parameters to [-1, 1].
# ``a`` is log-uniform, ``n`` is encoded through 1/n so that n = inf is finite.
_RE_RANGE = (1000.0, 3900.0)
_ALPHA_RANGE = (0.0, 45.0)
_LOG_A_RANGE = (math.log10(1.0 / 6.0), math.log10(1.5))


def _to_unit_interval(value: float, low: float, high: float) -> float:
    return 2.0 * (value - low) / (high - low) - 1.0


class ParamNormalizer:
    """Map selected PDE parameters of a case to a ``(P,)`` vector in ``[-1, 1]``.

    Args:
        names: Parameters to expose, in order (subset of ``Re, alpha, a, n``).
        enabled: When False the raw values are returned.
    """

    def __init__(self, names: Sequence[str], *, enabled: bool = True):
        unknown = [n for n in names if n not in ("Re", "alpha", "a", "n")]
        if unknown:
            raise ValueError(
                f"Unknown PDE parameter(s) {unknown}; expected subset of Re, alpha, a, n"
            )
        self.names = tuple(names)
        self.enabled = bool(enabled)

    @staticmethod
    def transform(name: str, value: float) -> float:
        """Normalize one parameter value to ``[-1, 1]``."""
        if name == "Re":
            return _to_unit_interval(value, *_RE_RANGE)
        if name == "alpha":
            return _to_unit_interval(value, *_ALPHA_RANGE)
        if name == "a":
            return _to_unit_interval(math.log10(value), *_LOG_A_RANGE)
        if name == "n":
            return 2.0 * (0.0 if math.isinf(value) else 1.0 / value) - 1.0
        raise KeyError(name)

    def __call__(self, params: CaseParams) -> np.ndarray:
        """Vector of (normalized) parameter values for ``params``; shape ``(P,)``."""
        values = params.as_dict()
        out = [
            self.transform(name, values[name]) if self.enabled else values[name]
            for name in self.names
        ]
        return np.asarray(out, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.names)
