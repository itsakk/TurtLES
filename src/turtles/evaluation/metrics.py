"""Rollout metrics (paper, Section 4).

For rollout step ``t`` with prediction ``û_t`` and reference ``u_t`` (both ``(N, C)``
in physical units) the per-channel errors are

* ``Rel-L2_c = ||û_c - u_c||_2 / (||u_c||_2 + eps)``
* ``VRMSE_c  = ||û_c - u_c||_2 / (||u_c - mean(u_c)||_2 + eps)``

and the headline numbers are the channel means, averaged over the steps of a
window (``1/T_out * sum_t``). Spectral band errors come from
:mod:`turtles.evaluation.spectra`. A :class:`RolloutRecord` keeps every per-frame
value so windows can be re-aggregated after the fact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

Window = tuple[int, int]


@dataclass(frozen=True)
class FrameErrors:
    """Per-channel errors of one frame (CPU float64 tensors of shape ``(C,)``)."""

    rel_l2: torch.Tensor
    vrmse: torch.Tensor

    @property
    def rel_l2_mean(self) -> float:
        """Channel mean of Rel-L2."""
        return float(self.rel_l2.mean())

    @property
    def vrmse_mean(self) -> float:
        """Channel mean of VRMSE."""
        return float(self.vrmse.mean())


@torch.no_grad()
def frame_errors(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> FrameErrors:
    """Per-channel Rel-L2 and VRMSE of ``pred`` against ``target`` (both ``(N, C)``)."""
    pred64 = pred.double()
    target64 = target.double()
    err = torch.linalg.vector_norm(pred64 - target64, dim=0)
    ref = torch.linalg.vector_norm(target64, dim=0)
    fluct = torch.linalg.vector_norm(target64 - target64.mean(dim=0, keepdim=True), dim=0)
    return FrameErrors(rel_l2=(err / (ref + eps)).cpu(), vrmse=(err / (fluct + eps)).cpu())


@dataclass
class RolloutRecord:
    """Per-frame history of one trajectory rollout (``T`` steps, ``C`` channels)."""

    channel_names: list[str]
    band_names: list[str] = field(default_factory=list)
    rel_l2: list[np.ndarray] = field(default_factory=list)  # each (C,)
    vrmse: list[np.ndarray] = field(default_factory=list)  # each (C,)
    spectral: list[np.ndarray] = field(default_factory=list)  # each (n_bands,)
    predict_seconds: list[float] = field(default_factory=list)
    non_finite_steps: list[int] = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        """Rollout steps recorded so far."""
        return len(self.rel_l2)

    def add(
        self,
        errors: FrameErrors | None,
        *,
        spectral: Sequence[float] | np.ndarray | None = None,
        predict_seconds: float = 0.0,
        finite: bool = True,
    ) -> None:
        """Append one step; a non-finite prediction is recorded as ``inf`` errors."""
        C = len(self.channel_names)
        if not finite or errors is None:
            self.non_finite_steps.append(self.num_steps)
            self.rel_l2.append(np.full(C, np.inf))
            self.vrmse.append(np.full(C, np.inf))
            if self.band_names:
                self.spectral.append(np.full(len(self.band_names), np.inf))
        else:
            self.rel_l2.append(errors.rel_l2.numpy().astype(np.float64))
            self.vrmse.append(errors.vrmse.numpy().astype(np.float64))
            if self.band_names:
                if spectral is None:
                    self.spectral.append(np.full(len(self.band_names), np.nan))
                else:
                    self.spectral.append(np.asarray(spectral, dtype=np.float64))
        self.predict_seconds.append(float(predict_seconds))

    def arrays(self) -> dict[str, np.ndarray]:
        """Stacked per-frame arrays (``rel_l2`` and ``vrmse`` are ``(T, C)``)."""
        out = {
            "rel_l2": np.stack(self.rel_l2)
            if self.rel_l2
            else np.zeros((0, len(self.channel_names))),
            "vrmse": np.stack(self.vrmse) if self.vrmse else np.zeros((0, len(self.channel_names))),
            "predict_seconds": np.asarray(self.predict_seconds, dtype=np.float64),
        }
        if self.band_names:
            out["spectral"] = (
                np.stack(self.spectral) if self.spectral else np.zeros((0, len(self.band_names)))
            )
        return out

    def save(self, path: str | Path) -> None:
        """Write the per-frame arrays to an ``.npz`` file."""
        np.savez(
            path,
            channel_names=np.array(self.channel_names),
            band_names=np.array(self.band_names),
            non_finite_steps=np.asarray(self.non_finite_steps, dtype=np.int64),
            **self.arrays(),
        )


def _window_mean(values: np.ndarray, lo: int, hi: int) -> float:
    """Mean of ``values`` over 1-indexed inclusive steps ``lo..hi`` (``nan`` if empty)."""
    block = values[max(lo - 1, 0) : hi]
    return float(np.mean(block)) if block.size else float("nan")


def summarize_record(
    record: RolloutRecord,
    *,
    cumulative_windows: Iterable[int],
    disjoint_windows: Iterable[Window] = (),
    divergence_threshold: float = 5.0,
    solver_gpu_seconds: float | None = None,
    num_gpus: int = 1,
) -> dict[str, float]:
    """Window averages, divergence flag and inference cost of one trajectory.

    Keys: ``rel_l2/0_k``, ``vrmse/0_k``, ``spec_<band>/0_k`` for cumulative windows
    (steps ``1..k``), ``*/win_lo_hi`` for disjoint windows, ``vrmse_channel/<c>`` over
    the longest cumulative window, ``diverged``, and ``time/*``.
    """
    arrays = record.arrays()
    T = record.num_steps
    rel_mean = arrays["rel_l2"].mean(axis=1) if T else np.zeros(0)
    vr_mean = arrays["vrmse"].mean(axis=1) if T else np.zeros(0)
    spectral = arrays.get("spectral")

    out: dict[str, float] = {}
    windows: list[tuple[str, int, int]] = [
        (f"0_{k}", 1, min(int(k), T)) for k in cumulative_windows
    ]
    windows += [(f"win_{lo}_{hi}", int(lo), min(int(hi), T)) for lo, hi in disjoint_windows]
    for tag, lo, hi in windows:
        if hi < lo:
            continue
        out[f"rel_l2/{tag}"] = _window_mean(rel_mean, lo, hi)
        out[f"vrmse/{tag}"] = _window_mean(vr_mean, lo, hi)
        if spectral is not None:
            for b, band in enumerate(record.band_names):
                out[f"spec_{band}/{tag}"] = _window_mean(spectral[:, b], lo, hi)

    longest = max((int(k) for k in cumulative_windows), default=T)
    hi = min(longest, T)
    for c, name in enumerate(record.channel_names):
        out[f"vrmse_channel/{name}"] = _window_mean(arrays["vrmse"][:, c], 1, hi)
        out[f"rel_l2_channel/{name}"] = _window_mean(arrays["rel_l2"][:, c], 1, hi)

    diverged = bool(record.non_finite_steps) or bool(
        T and np.nanmax(rel_mean) > divergence_threshold
    )
    out["diverged"] = float(diverged)

    times = arrays["predict_seconds"]
    if T:
        warm = times[1:] if T > 1 else times
        out["time/predict_per_frame_s"] = float(warm.mean())
        out["time/gpu_seconds"] = float(times.sum()) * num_gpus
        if solver_gpu_seconds:
            out["time/speedup_vs_solver"] = float(solver_gpu_seconds) / max(
                out["time/gpu_seconds"], 1e-9
            )
    return out


def aggregate_summaries(summaries: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Mean over trajectories of every key (``diverged`` becomes the diverged fraction)."""
    if not summaries:
        return {}
    keys = sorted({k for s in summaries for k in s})
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([s[key] for s in summaries if key in s], dtype=np.float64)
        out[key] = float(np.mean(values))
    out["num_trajectories"] = float(len(summaries))
    return out
