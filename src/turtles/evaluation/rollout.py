"""Autoregressive rollout of every trajectory of a split, scored frame by frame.

Per trajectory the evaluator streams frames from disk (multi-worker prefetch), rolls
the model out from the first frame(s) in normalized space, converts predictions and
references back to physical units and records Rel-L2, VRMSE, spanwise spectral band
errors and the inference time of every step. Results are written as::

    <output_dir>/metrics.json            aggregate over trajectories (+ per trajectory)
    <output_dir>/summary.md              human-readable headline numbers
    <output_dir>/trajectories/<case>/    per_frame.npz, spectra.npz, summary.json, plots
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from turtles.data.dataset import TurtlesData, build_loader
from turtles.data.io import Trajectory
from turtles.evaluation.metrics import (
    RolloutRecord,
    aggregate_summaries,
    frame_errors,
    summarize_record,
)
from turtles.evaluation.predict import predict_frame
from turtles.evaluation.spectra import (
    DEFAULT_BANDS,
    band_log_rmse,
    band_masks,
    build_spanwise_rings,
    spanwise_power,
)

logger = logging.getLogger(__name__)

_AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class PlotSettings:
    """Which figures to produce."""

    enabled: bool = True
    max_trajectories: int = 2
    channels: tuple[str, ...] = ("u",)
    grid_size: tuple[int, int] = (256, 256)
    snapshot_steps: tuple[int, ...] = (0, 1, 5, 10, 50, 100, 200, 398)


@dataclass
class RolloutSettings:
    """Rollout protocol and metric definitions (mirrors ``configs/eval.yaml``)."""

    split: str = "test"
    steps: int | None = 399
    max_trajectories: int | None = None
    query_chunk_size: int = 524_288
    input_subset_size: int | None = None
    context_frames: int = 1
    predict_residual: bool = True
    residual_scale: list[float] | None = None  # per-channel scale of the predicted increment
    amp: bool = True
    amp_dtype: str = "bfloat16"
    eps: float = 1e-8
    cumulative_windows: tuple[int, ...] = (1, 10, 20, 50, 100, 200, 399)
    disjoint_windows: tuple[tuple[int, int], ...] = ((1, 50), (51, 150), (151, 300), (301, 399))
    divergence_threshold: float = 5.0
    spectral_channel: str | None = "u"
    spectral_bands: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(DEFAULT_BANDS))
    solver_gpu_seconds: float | None = 13160.0
    loader_workers: int = 4
    loader_prefetch: int = 4
    plots: PlotSettings = field(default_factory=PlotSettings)


class RolloutEvaluator:
    """Roll a model out on the full mesh of every trajectory of a split."""

    def __init__(
        self,
        model: torch.nn.Module,
        data: TurtlesData,
        settings: RolloutSettings,
        *,
        output_dir: str | Path,
        device: torch.device,
    ):
        self.model = model.to(device).eval()
        self.data = data
        self.s = settings
        self.output_dir = Path(output_dir)
        self.device = device
        self.amp_enabled = bool(settings.amp) and device.type == "cuda"
        self.amp_dtype = _AMP_DTYPES[settings.amp_dtype]
        self.channel_names = list(data.fields)
        self.band_names = list(settings.spectral_bands) if settings.spectral_channel else []

    # ------------------------------------------------------------------ helpers

    def _frame_loader(self, trajectory: Trajectory, num_frames: int):
        frame_ds = self.data.frame_dataset(trajectory, start=0, end=num_frames)
        context = None
        if self.s.loader_workers > 0:
            # Workers must not inherit CUDA state: spawn them from a clean process.
            methods = mp.get_all_start_methods()
            context = mp.get_context("forkserver" if "forkserver" in methods else "spawn")
        loader = build_loader(
            frame_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.s.loader_workers,
            prefetch_factor=self.s.loader_prefetch,
            pin_memory=self.device.type == "cuda",
            persistent_workers=False,
            multiprocessing_context=context,
        )
        return frame_ds, loader

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------ one trajectory

    @torch.no_grad()
    def rollout_trajectory(
        self, trajectory: Trajectory, *, traj_dir: Path | None = None, make_plots: bool = False
    ) -> tuple[RolloutRecord, dict[str, Any]]:
        """Roll out one trajectory; returns its record and summary (also written to ``traj_dir``)."""
        s = self.s
        T_in = s.context_frames
        available = trajectory.num_frames - T_in
        steps = available if s.steps is None else min(int(s.steps), available)
        if steps <= 0:
            raise ValueError(
                f"{trajectory.name} has too few frames ({trajectory.num_frames}) for a rollout"
            )

        frame_ds, loader = self._frame_loader(trajectory, T_in + steps)
        frames = iter(loader)
        normalizer = self.data.field_normalizer
        coords = frame_ds.coords().to(self.device).unsqueeze(0)  # (1, N, 3)
        params = self.data.params_tensor(trajectory)
        params = None if params is None else params.to(self.device)

        window = (
            torch.stack([next(frames)[0] for _ in range(T_in)], dim=0).unsqueeze(0).to(self.device)
        )
        record = RolloutRecord(self.channel_names, self.band_names)

        rings = masks = None
        spec_channel = None
        spec_sum_ref = spec_sum_pred = spec_sum_log = None
        if s.spectral_channel:
            spec_channel = self.channel_names.index(s.spectral_channel)
            try:
                rings = build_spanwise_rings(coords[0])
                masks = band_masks(rings.wavenumbers, s.spectral_bands)
                K = rings.wavenumbers.numel()
                spec_sum_ref = torch.zeros(K, dtype=torch.float64)
                spec_sum_pred = torch.zeros(K, dtype=torch.float64)
                spec_sum_log = torch.zeros(K, dtype=torch.float64)
            except RuntimeError as exc:
                logger.warning(
                    "%s: spanwise rings unavailable (%s); spectral metrics skipped",
                    trajectory.name,
                    exc,
                )

        plots = s.plots
        snapshot_steps = (
            sorted({int(t) for t in plots.snapshot_steps if 0 <= int(t) < steps})
            if make_plots
            else []
        )
        projection = None
        snaps_ref: dict[str, list[np.ndarray]] = {c: [] for c in plots.channels}
        snaps_pred: dict[str, list[np.ndarray]] = {c: [] for c in plots.channels}
        if make_plots and plots.enabled:
            from turtles.evaluation.slices import apply_grid_projection, build_grid_projection

            projection = build_grid_projection(coords[0].cpu(), grid_size=tuple(plots.grid_size))

        for t in range(steps):
            target = next(frames)[0].to(self.device, non_blocking=True).unsqueeze(0)  # (1, N, C)
            last = window[:, -1]
            self._sync()
            tic = time.perf_counter()
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                pred = predict_frame(
                    self.model, window, coords, params,
                    query_chunk_size=s.query_chunk_size, input_subset_size=s.input_subset_size,
                )  # fmt: skip
            pred = pred.float()
            if s.predict_residual:
                if s.residual_scale is not None:
                    pred = pred * torch.tensor(
                        s.residual_scale, device=pred.device, dtype=pred.dtype
                    )
                pred = pred + last
            self._sync()
            predict_seconds = time.perf_counter() - tic

            finite = bool(torch.isfinite(pred).all())
            pred_phys = normalizer.unnormalize(pred[0])
            target_phys = normalizer.unnormalize(target[0].float())
            spectral = None
            if finite:
                errors = frame_errors(pred_phys, target_phys, eps=s.eps)
                if rings is not None and masks is not None and spec_channel is not None:
                    p_pred = spanwise_power(pred_phys[:, spec_channel], rings)
                    p_ref = spanwise_power(target_phys[:, spec_channel], rings)
                    bands = band_log_rmse(p_pred, p_ref, masks)
                    spectral = [bands[name] for name in self.band_names]
                    spec_sum_ref += p_ref.double().cpu()
                    spec_sum_pred += p_pred.double().cpu()
                    spec_sum_log += (
                        (p_pred.double() + 1e-12).log() - (p_ref.double() + 1e-12).log()
                    ).cpu()
            else:
                errors = None
            record.add(errors, spectral=spectral, predict_seconds=predict_seconds, finite=finite)

            if projection is not None and t in snapshot_steps:
                for name in plots.channels:
                    c = self.channel_names.index(name)
                    snaps_ref[name].append(
                        apply_grid_projection(target_phys[:, c : c + 1], projection)[..., 0]
                        .cpu()
                        .numpy()
                    )
                    snaps_pred[name].append(
                        apply_grid_projection(pred_phys[:, c : c + 1], projection)[..., 0]
                        .cpu()
                        .numpy()
                    )

            # Autoregression continues in normalized space, like training.
            window = torch.cat([window[:, 1:], pred.to(window.dtype).unsqueeze(1)], dim=1)
            if (t + 1) % 50 == 0 and t + 1 < steps:
                vrmse = float("nan") if errors is None else float(errors.vrmse.mean())
                logger.info(
                    "  %s: step %d/%d, %.2f s/step, vrmse %.4g",
                    trajectory.name,
                    t + 1,
                    steps,
                    predict_seconds,
                    vrmse,
                )

        summary = summarize_record(
            record,
            cumulative_windows=s.cumulative_windows,
            disjoint_windows=s.disjoint_windows,
            divergence_threshold=s.divergence_threshold,
            solver_gpu_seconds=s.solver_gpu_seconds,
        )
        summary["num_points"] = float(trajectory.num_points)
        summary["rollout_steps"] = float(record.num_steps)

        n_ok = record.num_steps - len(record.non_finite_steps)
        if traj_dir is not None:
            traj_dir.mkdir(parents=True, exist_ok=True)
            record.save(traj_dir / "per_frame.npz")
            (traj_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            if rings is not None and n_ok:
                np.savez(
                    traj_dir / "spectra.npz",
                    wavenumbers=rings.wavenumbers.numpy(),
                    power_ref=(spec_sum_ref / n_ok).numpy(),
                    power_pred=(spec_sum_pred / n_ok).numpy(),
                    log_discrepancy=(spec_sum_log / n_ok).numpy(),
                    channel=np.array(s.spectral_channel),
                )
            if make_plots and plots.enabled:
                self._write_plots(
                    traj_dir, record, snapshot_steps, snaps_ref, snaps_pred, projection,
                    rings, n_ok, spec_sum_ref, spec_sum_pred, spec_sum_log,
                )  # fmt: skip
        return record, summary

    def _write_plots(
        self,
        traj_dir,
        record,
        snapshot_steps,
        snaps_ref,
        snaps_pred,
        projection,
        rings,
        n_ok,
        sum_ref,
        sum_pred,
        sum_log,
    ) -> None:
        from turtles.evaluation.plots import plot_error_curves, plot_snapshots, plot_spectra

        arrays = record.arrays()
        if arrays["rel_l2"].shape[0]:
            plot_error_curves(
                arrays["rel_l2"],
                arrays["vrmse"],
                self.channel_names,
                path=traj_dir / "error_curves.pdf",
            )
        if projection is not None and snapshot_steps:
            for name in self.s.plots.channels:
                if snaps_ref[name]:
                    plot_snapshots(
                        np.stack(snaps_ref[name]), np.stack(snaps_pred[name]), snapshot_steps,
                        title=f"{name} at z = {projection.z0:.2f}", path=traj_dir / f"snapshots_{name}.pdf",
                        extent=projection.extent,
                    )  # fmt: skip
        if rings is not None and n_ok:
            plot_spectra(
                rings.wavenumbers.numpy(), (sum_ref / n_ok).numpy(), (sum_pred / n_ok).numpy(),
                (sum_log / n_ok).numpy(), label="prediction", path=traj_dir / "spectra.pdf",
            )  # fmt: skip

    # ------------------------------------------------------------------ driver

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        """Roll out every trajectory of the split and write the aggregate report."""
        trajectories = self.data.trajectories(self.s.split)
        if self.s.max_trajectories is not None:
            trajectories = trajectories[: int(self.s.max_trajectories)]
        if not trajectories:
            raise RuntimeError(
                f"No trajectory found for split '{self.s.split}' in {self.data.split_dir(self.s.split)}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Evaluating %d trajectories of split '%s' -> %s",
            len(trajectories),
            self.s.split,
            self.output_dir,
        )

        per_trajectory: dict[str, dict[str, float]] = {}
        tic = time.perf_counter()
        for i, trajectory in enumerate(trajectories):
            traj_dir = self.output_dir / "trajectories" / trajectory.name
            _, summary = self.rollout_trajectory(
                trajectory, traj_dir=traj_dir, make_plots=i < self.s.plots.max_trajectories
            )
            per_trajectory[trajectory.name] = summary
            headline = {
                k: v
                for k, v in summary.items()
                if k in ("rel_l2/0_1", "vrmse/0_1", "diverged", "time/gpu_seconds")
            }
            logger.info(
                "[%d/%d] %s (N=%d): %s",
                i + 1,
                len(trajectories),
                trajectory.name,
                trajectory.num_points,
                " ".join(f"{k}={v:.4g}" for k, v in headline.items()),
            )

        aggregate = aggregate_summaries(list(per_trajectory.values()))
        aggregate["time/eval_wall_s"] = time.perf_counter() - tic
        report = {
            "aggregate": aggregate,
            "per_trajectory": per_trajectory,
            "settings": _settings_dict(self.s),
        }
        (self.output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
        (self.output_dir / "summary.md").write_text(render_summary(aggregate, self.band_names))
        logger.info("Wrote %s", self.output_dir / "metrics.json")
        return report


def _settings_dict(settings: RolloutSettings) -> dict[str, Any]:
    import dataclasses

    out = dataclasses.asdict(settings)
    return json.loads(json.dumps(out, default=str))


def render_summary(aggregate: dict[str, float], band_names: list[str]) -> str:
    """Markdown with the next-frame and rollout-window headline numbers."""
    lines = ["# Rollout summary", ""]
    lines.append(
        f"Trajectories: {int(aggregate.get('num_trajectories', 0))}, diverged fraction: {aggregate.get('diverged', float('nan')):.2f}"
    )
    lines += ["", "| window | Rel-L2 | VRMSE |" + "".join(f" spec {b} |" for b in band_names)]
    lines.append("|---|---|---|" + "---|" * len(band_names))
    tags = sorted({k.split("/", 1)[1] for k in aggregate if k.startswith("vrmse/")}, key=_tag_key)
    for tag in tags:
        row = f"| {tag} | {aggregate.get(f'rel_l2/{tag}', float('nan')):.4g} | {aggregate.get(f'vrmse/{tag}', float('nan')):.4g} |"
        row += "".join(
            f" {aggregate.get(f'spec_{b}/{tag}', float('nan')):.4g} |" for b in band_names
        )
        lines.append(row)
    if "time/gpu_seconds" in aggregate:
        lines += ["", f"GPU-seconds per trajectory: {aggregate['time/gpu_seconds']:.1f}"]
        if "time/speedup_vs_solver" in aggregate:
            lines.append(f"Speed-up versus solver: {aggregate['time/speedup_vs_solver']:.1f}x")
    return "\n".join(lines) + "\n"


def _tag_key(tag: str) -> tuple[int, int]:
    parts = tag.replace("win_", "").split("_")
    try:
        return (0 if tag.startswith("0_") else 1, int(parts[-1]))
    except ValueError:
        return (2, 0)
