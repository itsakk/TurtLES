"""Compute the per-field statistics (``stats.yaml``) of a set of trajectories.

The dataset ships with statistics of the train split. This module recomputes them
for custom data generated with :mod:`turtles.generation` or for subsets.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from turtles.data.io import FIELDS, Trajectory, read_frame

STAT_KEYS: tuple[str, ...] = ("mean", "std", "rms", "mean_delta", "rms_delta", "std_delta")


class _Moments:
    """Running first and second moments per channel in float64."""

    def __init__(self, num_channels: int):
        self.sum = np.zeros(num_channels, dtype=np.float64)
        self.sum_sq = np.zeros(num_channels, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        x64 = x.astype(np.float64, copy=False)
        self.sum += x64.sum(axis=0)
        self.sum_sq += np.square(x64).sum(axis=0)
        self.count += x64.shape[0]

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = max(self.count, 1)
        mean = self.sum / n
        rms = np.sqrt(self.sum_sq / n)
        std = np.sqrt(np.maximum(self.sum_sq / n - mean**2, 0.0))
        return mean, std, rms


def compute_stats(
    trajectories: Sequence[Trajectory],
    fields: Sequence[str] = FIELDS,
    *,
    frame_stride: int = 10,
    max_nodes: int | None = 1_000_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Mean / std / rms of the fields and of their frame-to-frame increments.

    Args:
        trajectories: Trajectories to accumulate over (typically the train split).
        fields: Field names in channel order.
        frame_stride: Use every ``frame_stride``-th frame (and its successor).
        max_nodes: Random node subset per trajectory (``None`` = all nodes).
        seed: Seed of the node subset.
    """
    rng = np.random.default_rng(seed)
    states = _Moments(len(fields))
    deltas = _Moments(len(fields))
    for traj in trajectories:
        sel = None
        if max_nodes is not None and traj.num_points > max_nodes:
            sel = np.sort(rng.choice(traj.num_points, max_nodes, replace=False))
        for t in range(0, traj.num_frames - 1, frame_stride):
            current = read_frame(traj, t, fields, sel)
            nxt = read_frame(traj, t + 1, fields, sel)
            states.update(current)
            deltas.update(nxt - current)
    mean, std, rms = states.finalize()
    mean_d, std_d, rms_d = deltas.finalize()
    values = {
        "mean": mean,
        "std": std,
        "rms": rms,
        "mean_delta": mean_d,
        "rms_delta": rms_d,
        "std_delta": std_d,
    }
    return {
        key: {field: float(f"{vec[i]:.5g}") for i, field in enumerate(fields)}
        for key, vec in values.items()
    }


def write_stats(stats: dict[str, dict[str, float]], path: str | Path) -> Path:
    """Write statistics to ``path`` in the ``stats.yaml`` layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        yaml.safe_dump({k: stats[k] for k in STAT_KEYS if k in stats}, handle, sort_keys=False)
    return path
