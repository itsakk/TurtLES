"""Step-based trainer for the single-step forecasting task.

The model learns ``u_t -> u_{t+1}`` (optionally as the increment ``u_{t+1} - u_t``) on
random node subsets. The loop handles DDP, bf16 autocast, gradient accumulation,
the LR schedule, step-based validation, logging and resumable checkpoints.

Conventions:

* Per-step metrics are kept as detached tensors and converted to floats only when
  logs are flushed, so the hot path never synchronizes with the device.
* ``last.pt`` is written every ``save_every`` steps and at the end; ``best.pt`` tracks
  the lowest validation loss (or training loss without a validation set).
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from turtles.tracking import MeterDict, WandbLogger
from turtles.training.checkpoint import (
    BEST_NAME,
    LAST_NAME,
    load_checkpoint,
    load_model_weights,
    save_checkpoint,
    unwrap,
)
from turtles.training.distributed import DistInfo

logger = logging.getLogger(__name__)

_AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _grad_scaler(device_type: str, enabled: bool):
    if hasattr(torch.amp, "GradScaler"):  # torch >= 2.3
        return torch.amp.GradScaler(device_type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@dataclass
class TrainerConfig:
    """Everything the loop needs besides the model, loaders and optimizer."""

    steps: int
    context_frames: int = 1
    predict_residual: bool = True
    train_query_points: int | None = None
    # Per-channel scale of the increment target (C,), see TurtlesData.increment_scale.
    residual_scale: list[float] | None = None
    accumulate_gradients: int = 1
    clip_grad_norm: float = 1.0
    amp: bool = True
    amp_dtype: str = "bfloat16"
    log_every: int = 100
    save_every: int = 2000
    val_every: int = 5000
    val_max_batches: int | None = 50
    ddp_find_unused_parameters: bool = False


class Trainer:
    """Train a forecaster; see the module docstring for the conventions."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        config: TrainerConfig,
        run_dir: str | Path,
        device: torch.device,
        dist: DistInfo,
        channel_names: Sequence[str] | None = None,
        tracker: WandbLogger | None = None,
        checkpoint_extra: dict[str, Any] | None = None,
    ):
        self.cfg = config
        self.device = device
        self.dist = dist
        self.run_dir = Path(run_dir)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tracker = tracker or WandbLogger(enabled=False)
        self.channel_names = list(channel_names) if channel_names else None
        self.checkpoint_extra = dict(checkpoint_extra or {})

        model = model.to(device)
        if dist.distributed and any(p.requires_grad for p in model.parameters()):
            self.model: torch.nn.Module = DDP(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=config.ddp_find_unused_parameters,
            )
        else:
            self.model = model
        self.raw_model = unwrap(self.model)
        self.supports_query_coords = bool(getattr(self.raw_model, "supports_query_coords", False))

        self.amp_dtype = _AMP_DTYPES[config.amp_dtype]
        self.amp_enabled = bool(config.amp) and device.type == "cuda"
        self.scaler = _grad_scaler(
            device.type, enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )

        self.residual_scale = (
            None
            if config.residual_scale is None
            else torch.tensor(config.residual_scale, dtype=torch.float32, device=device)
        )
        self.step = 0
        self.best_metric = math.inf
        self._last_saved_step = -1
        self._pending: list[dict[str, torch.Tensor]] = []
        self._meters = MeterDict()

    # ------------------------------------------------------------------ loss

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled)

    def compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """MSE on the (normalized) target; metrics are detached tensors."""
        fields = batch["fields"].to(self.device, non_blocking=True)  # (B, T, N, C)
        coords = batch["coords"].to(self.device, non_blocking=True)  # (B, N, 3)
        params = batch.get("params")
        params = (
            params.to(self.device, non_blocking=True)
            if params is not None and params.numel()
            else None
        )

        T_in = self.cfg.context_frames
        context = fields[:, :T_in]
        target = fields[:, T_in]
        last = context[:, -1]
        regression_target = target - last if self.cfg.predict_residual else target
        if self.cfg.predict_residual and self.residual_scale is not None:
            regression_target = regression_target / self.residual_scale

        nq = self.cfg.train_query_points
        N = coords.shape[1]
        if self.supports_query_coords and nq is not None and nq < N:
            qi = torch.randperm(N, device=self.device)[:nq]
            pred = self.model(context, coords, params, query_coords=coords[:, qi])
            regression_target, last, target = regression_target[:, qi], last[:, qi], target[:, qi]
        else:
            pred = self.model(context, coords, params)
        loss = F.mse_loss(pred.float(), regression_target.float())

        with torch.no_grad():
            if self.cfg.predict_residual:
                increment = pred.float()
                if self.residual_scale is not None:
                    increment = increment * self.residual_scale
                reconstructed = increment + last.float()
            else:
                reconstructed = pred.float()
            sq_err = (reconstructed - target.float()).square()
            metrics = {"rmse": sq_err.mean().sqrt()}
            # Reference: the persistence predictor (zero increment) on the same batch. A
            # model has learnt something only once `loss` drops below `loss_persistence`;
            # `skill` = 1 - loss / loss_persistence is positive in that case.
            persistence_increment = target.float() - last.float()
            if self.residual_scale is not None:
                persistence_increment = persistence_increment / self.residual_scale
            persistence_loss = persistence_increment.square().mean()
            metrics["loss_persistence"] = persistence_loss
            metrics["skill"] = 1.0 - loss.detach() / persistence_loss.clamp_min(1e-30)
            # Collapse diagnostic: a model predicting the mean increment has pred_scale ~ 0.
            metrics["pred_scale"] = (
                pred.detach().float().std() / regression_target.float().std().clamp_min(1e-30)
            )
            if self.channel_names and sq_err.shape[-1] == len(self.channel_names):
                err_c = (
                    (pred.detach().float() - regression_target.float()).square().mean(dim=(0, 1))
                )
                base_c = persistence_increment.square().mean(dim=(0, 1)).clamp_min(1e-30)
                for i, name in enumerate(self.channel_names):
                    metrics[f"skill_channel/{name}"] = 1.0 - err_c[i] / base_c[i]
            if self.channel_names and sq_err.shape[-1] == len(self.channel_names):
                per_channel = sq_err.mean(dim=(0, 1)).sqrt()
                for i, name in enumerate(self.channel_names):
                    metrics[f"rmse_channel/{name}"] = per_channel[i]
        return loss, metrics

    # ------------------------------------------------------------------ loop

    def _batches(self) -> Iterator[dict[str, Any]]:
        sampler = getattr(self.train_loader, "sampler", None)
        epoch = self.step // max(len(self.train_loader), 1)
        while True:
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            yield from self.train_loader
            epoch += 1

    def fit(self) -> float:
        """Run until ``cfg.steps``; returns the best validation (or training) loss."""
        cfg = self.cfg
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batches = self._batches()
        tic = time.perf_counter()
        while self.step < cfg.steps:
            batch = next(batches)
            self.step += 1
            self._train_step(batch, boundary=(self.step % cfg.accumulate_gradients == 0))

            if self.step % cfg.log_every == 0:
                elapsed = time.perf_counter() - tic
                self._flush_train_metrics(steps_per_second=cfg.log_every / max(elapsed, 1e-9))
                tic = time.perf_counter()
            if cfg.val_every and self.step % cfg.val_every == 0 and self.val_loader is not None:
                self._run_validation()
            if cfg.save_every and self.step % cfg.save_every == 0:
                self.save(LAST_NAME)

        if self._pending:
            self._flush_train_metrics()
        if self.val_loader is not None and cfg.val_every and self.step % cfg.val_every != 0:
            self._run_validation()
        if self._last_saved_step != self.step:
            self.save(LAST_NAME)
        return self.best_metric

    def _train_step(self, batch: dict[str, Any], *, boundary: bool) -> None:
        sync = contextlib.nullcontext()
        if isinstance(self.model, DDP) and not boundary:
            sync = self.model.no_sync()
        with sync:
            with self._autocast():
                loss, metrics = self.compute_loss(batch)
            self.scaler.scale(loss / self.cfg.accumulate_gradients).backward()
        if boundary:
            if self.cfg.clip_grad_norm and self.cfg.clip_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        metrics["loss"] = loss.detach()
        self._pending.append(metrics)

    def _flush_train_metrics(self, steps_per_second: float | None = None) -> None:
        for metrics in self._pending:
            self._meters.update(metrics)
        self._pending.clear()
        self._meters.sync()
        averages = self._meters.averages()
        self._meters.reset()
        lr = self.optimizer.param_groups[0]["lr"]
        if self.dist.is_main:
            body = " ".join(f"{k}={v:.4e}" for k, v in averages.items() if "/" not in k)
            rate = f" {steps_per_second:.2f} it/s" if steps_per_second else ""
            logger.info("step %d/%d | %s lr=%.2e%s", self.step, self.cfg.steps, body, lr, rate)
            payload = {f"train/{k}": v for k, v in averages.items()}
            payload["train/lr"] = lr
            if steps_per_second:
                payload["train/steps_per_second"] = steps_per_second
            self.tracker.log(self.step, payload)
        if self.val_loader is None:
            self._update_best(averages.get("loss", math.inf))

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Average loss and metrics over (up to ``val_max_batches``) validation batches."""
        assert self.val_loader is not None
        self.model.eval()
        meters = MeterDict()
        for i, batch in enumerate(self.val_loader):
            if self.cfg.val_max_batches is not None and i >= self.cfg.val_max_batches:
                break
            with self._autocast():
                loss, metrics = self.compute_loss(batch)
            metrics["loss"] = loss.detach()
            meters.update(metrics)
        meters.sync()
        self.model.train()
        return meters.averages()

    def _run_validation(self) -> None:
        results = self.validate()
        if self.dist.is_main:
            body = " ".join(f"{k}={v:.4e}" for k, v in results.items() if "/" not in k)
            logger.info("step %d | val: %s", self.step, body)
            self.tracker.log(self.step, {f"val/{k}": v for k, v in results.items()})
        self._update_best(results.get("loss", math.inf))

    def _update_best(self, metric: float) -> None:
        if metric < self.best_metric:
            self.best_metric = float(metric)
            self.save(BEST_NAME)

    # ------------------------------------------------------------------ checkpoints

    def state(self) -> dict[str, Any]:
        """Everything needed to resume (plus the extras passed at construction)."""
        return {
            "step": self.step,
            "best_metric": self.best_metric,
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            **self.checkpoint_extra,
        }

    def save(self, name: str) -> Path | None:
        """Write ``<run_dir>/<name>`` on the main rank."""
        if not self.dist.is_main:
            return None
        path = save_checkpoint(self.run_dir / name, self.state())
        if name == LAST_NAME:
            self._last_saved_step = self.step
        logger.info("Saved %s (step %d)", path, self.step)
        return path

    def load(self, path: str | Path) -> None:
        """Resume model, optimizer, scheduler and counters from ``path``."""
        ckpt = load_checkpoint(path, map_location=self.device)
        load_model_weights(self.raw_model, ckpt["model"])
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.step = int(ckpt.get("step", 0))
        self.best_metric = float(ckpt.get("best_metric", math.inf))
        logger.info("Resumed from %s at step %d", path, self.step)
