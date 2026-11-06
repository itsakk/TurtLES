"""Turn a resolved Hydra config into a training run or a rollout evaluation.

The command-line entry points (:mod:`turtles.cli`) are thin wrappers around
:func:`run_training` and :func:`run_evaluation`, which are also usable from Python::

    from omegaconf import OmegaConf
    from turtles.experiment import load_config, run_training

    cfg = load_config("train", overrides=["experiment=debug", "data.source=hf"])
    run_training(cfg)
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from turtles.data.dataset import TurtlesData, build_loader
from turtles.data.normalization import FieldNormalizer
from turtles.evaluation.rollout import PlotSettings, RolloutEvaluator, RolloutSettings
from turtles.models import build_model, count_parameters
from turtles.tracking import WandbLogger
from turtles.training.checkpoint import (
    LAST_NAME,
    load_checkpoint,
    load_model_weights,
    resolve_checkpoint,
)
from turtles.training.distributed import DistInfo, cleanup_distributed, init_distributed
from turtles.training.schedule import build_lr_scheduler, build_optimizer
from turtles.training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

if not OmegaConf.has_resolver("len"):
    OmegaConf.register_new_resolver("len", len)


def load_config(name: str, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose ``configs/<name>.yaml`` with Hydra overrides, outside of a Hydra app."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=name, overrides=list(overrides))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_run_name(cfg: DictConfig) -> str:
    """``cfg.run_name`` or ``<jobid>-<model>-<subset>`` (SLURM) / ``<model>-<subset>-<time>``."""
    if cfg.get("run_name"):
        return str(cfg.run_name)
    model = str(cfg.model.name)
    subset = str(cfg.data.get("subset", "full"))
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"{job_id}-{model}-{subset}"
    return f"{model}-{subset}-{time.strftime('%Y%m%d-%H%M%S')}"


def _as_dict(cfg: DictConfig | Any) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def fixed_subset(
    dataset: torch.utils.data.Dataset, size: int, seed: int
) -> torch.utils.data.Dataset:
    """Deterministic random subset of ``size`` windows spanning all trajectories.

    Validation on the first windows of a split would score a single trajectory; a seeded
    random subset is representative and identical at every validation pass.
    """
    n = len(dataset)  # type: ignore[arg-type]
    if size >= n:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator)[:size].sort().values.tolist()
    return torch.utils.data.Subset(dataset, indices)


def _format_count(n: int) -> str:
    return f"{n / 1e6:.2f} M" if n >= 1e6 else f"{n / 1e3:.1f} K" if n >= 1e3 else str(n)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_training(cfg: DictConfig) -> float:
    """Train the configured model; returns the best validation (or training) loss."""
    dist, device = init_distributed(use_cuda=True)
    seed_everything(int(cfg.seed) + dist.rank)
    if device.type == "cuda":
        torch.set_float32_matmul_precision(str(cfg.precision.matmul_precision))

    window = int(cfg.task.context_frames) + 1
    data = TurtlesData(cfg.data, splits=("train", "val"))
    data.check_batchable("train", int(cfg.optim.batch_size))
    train_ds = data.window_dataset("train", window=window)
    if train_ds is None:
        raise RuntimeError(f"No training trajectory under {data.split_dir('train')}")
    val_ds = data.window_dataset("val", window=window) if int(cfg.validation.every) > 0 else None
    if val_ds is not None and cfg.validation.get("max_batches"):
        val_size = int(cfg.validation.max_batches) * int(cfg.optim.batch_size) * dist.world_size
        val_ds = fixed_subset(val_ds, val_size, seed=int(cfg.seed))

    loader_kwargs = dict(
        num_workers=int(cfg.loader.num_workers),
        prefetch_factor=int(cfg.loader.prefetch_factor),
        pin_memory=bool(cfg.loader.pin_memory) and device.type == "cuda",
        persistent_workers=bool(cfg.loader.persistent_workers),
        distributed=dist.distributed,
        seed=int(cfg.seed),
    )
    train_loader = build_loader(
        train_ds,
        batch_size=int(cfg.optim.batch_size),
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = (
        build_loader(val_ds, batch_size=int(cfg.optim.batch_size), shuffle=False, **loader_kwargs)
        if val_ds is not None
        else None
    )

    model = build_model(cfg.model)
    num_params = count_parameters(model)
    run_name = resolve_run_name(cfg)
    run_dir = Path(str(cfg.output_root)).expanduser() / "runs" / run_name
    resolved = _as_dict(cfg)
    resolved["run_name"] = run_name
    if dist.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(resolved), run_dir / "config.yaml")

    if bool(cfg.precision.compile) and device.type == "cuda":
        model = torch.compile(model)

    accumulate = int(cfg.optim.accumulate_gradients)
    optimizer = build_optimizer(
        model,
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        betas=tuple(cfg.optim.betas),
    )
    scheduler = build_lr_scheduler(
        optimizer,
        total_steps=max(int(cfg.optim.steps) // accumulate, 1),
        warmup_steps=int(cfg.optim.warmup_steps) // accumulate,
        schedule=str(cfg.optim.schedule),
        lr_min_ratio=float(cfg.optim.lr_min) / float(cfg.optim.lr),
    )

    wandb_cfg = cfg.logging.wandb
    tracker = WandbLogger(
        enabled=bool(wandb_cfg.enabled) and dist.is_main,
        project=str(wandb_cfg.project),
        entity=wandb_cfg.entity,
        mode=str(wandb_cfg.mode),
        directory=str(wandb_cfg.dir),
        run_name=run_name,
        config=resolved,
        tags=[str(t) for t in (wandb_cfg.tags or [])],
    )

    residual_scale = None
    if bool(cfg.task.predict_residual) and bool(cfg.task.get("scale_residual", False)):
        residual_scale = data.increment_scale().tolist()
    trainer_cfg = TrainerConfig(
        steps=int(cfg.optim.steps),
        context_frames=int(cfg.task.context_frames),
        predict_residual=bool(cfg.task.predict_residual),
        train_query_points=cfg.task.get("train_query_points"),
        residual_scale=residual_scale,
        accumulate_gradients=accumulate,
        clip_grad_norm=float(cfg.optim.clip_grad_norm),
        amp=bool(cfg.precision.amp),
        amp_dtype=str(cfg.precision.amp_dtype),
        log_every=int(cfg.logging.every),
        save_every=int(cfg.checkpoint.save_every),
        val_every=int(cfg.validation.every),
        val_max_batches=cfg.validation.get("max_batches"),
        ddp_find_unused_parameters=bool(cfg.precision.ddp_find_unused_parameters),
    )
    trainer = Trainer(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=trainer_cfg,
        run_dir=run_dir,
        device=device,
        dist=dist,
        channel_names=data.fields,
        tracker=tracker,
        checkpoint_extra={
            "config": resolved,
            "normalizer": data.field_normalizer.state_dict(),
            "residual_scale": residual_scale,
            "model_name": str(cfg.model.name),
        },
    )

    resume = cfg.get("resume")
    if resume:
        resume_path = run_dir / LAST_NAME if str(resume).lower() == "true" else Path(str(resume))
        trainer.load(resume_path)

    if dist.is_main:
        _log_training_banner(cfg, data, train_ds, val_ds, num_params, run_dir, dist, device)
    try:
        best = trainer.fit()
    finally:
        tracker.finish()
        cleanup_distributed()
    return best


def _log_training_banner(
    cfg, data, train_ds, val_ds, num_params, run_dir, dist: DistInfo, device
) -> None:
    effective_batch = (
        int(cfg.optim.batch_size) * int(cfg.optim.accumulate_gradients) * dist.world_size
    )
    lines = [
        "=" * 72,
        f" TurtLES training run: {run_dir.name}",
        "=" * 72,
        f" run dir       : {run_dir}",
        f" model         : {cfg.model.name}  ({_format_count(num_params)} trainable parameters)",
        f" task          : context={cfg.task.context_frames} residual={cfg.task.predict_residual} "
        f"scale_residual={cfg.task.get('scale_residual', False)} "
        f"query_points={cfg.task.get('train_query_points') or 'all'}",
        f" data          : {data.summary()}",
        f"   trajectories: train={len(data.trajectories('train'))} val={len(data.trajectories('val'))}",
        f"   windows     : train={len(train_ds):,}" + (f" val={len(val_ds):,}" if val_ds else ""),
        f" device        : {device} x {dist.world_size}  amp={cfg.precision.amp}/{cfg.precision.amp_dtype} "
        f"compile={cfg.precision.compile}",
        f" batch         : {cfg.optim.batch_size} x accumulate {cfg.optim.accumulate_gradients} x "
        f"{dist.world_size} GPU(s) = {effective_batch}",
        f" optimizer     : AdamW lr={float(cfg.optim.lr):g} -> {float(cfg.optim.lr_min):g} "
        f"({cfg.optim.schedule}, warmup {cfg.optim.warmup_steps}), wd={float(cfg.optim.weight_decay):g}",
        f" steps         : {int(cfg.optim.steps):,}  (val every {cfg.validation.every}, "
        f"checkpoint every {cfg.checkpoint.save_every})",
        "=" * 72,
    ]
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _rollout_settings(
    cfg: DictConfig, *, context_frames: int, predict_residual: bool
) -> RolloutSettings:
    spectral = cfg.metrics.spectral
    plots = cfg.plots
    return RolloutSettings(
        split=str(cfg.rollout.split),
        steps=cfg.rollout.get("steps"),
        max_trajectories=cfg.rollout.get("max_trajectories"),
        query_chunk_size=int(cfg.rollout.query_chunk_size),
        input_subset_size=cfg.rollout.get("input_subset_size"),
        context_frames=context_frames,
        predict_residual=predict_residual,
        amp=bool(cfg.precision.amp),
        amp_dtype=str(cfg.precision.amp_dtype),
        eps=float(cfg.metrics.eps),
        cumulative_windows=tuple(int(k) for k in cfg.metrics.cumulative_windows),
        disjoint_windows=tuple((int(lo), int(hi)) for lo, hi in cfg.metrics.disjoint_windows),
        divergence_threshold=float(cfg.metrics.divergence_threshold),
        spectral_channel=spectral.get("channel"),
        spectral_bands={str(k): (int(v[0]), int(v[1])) for k, v in spectral.bands.items()},
        solver_gpu_seconds=cfg.metrics.get("solver_gpu_seconds"),
        loader_workers=int(cfg.loader.num_workers),
        loader_prefetch=int(cfg.loader.prefetch_factor),
        plots=PlotSettings(
            enabled=bool(plots.enabled),
            max_trajectories=int(plots.max_trajectories),
            channels=tuple(str(c) for c in plots.channels),
            grid_size=tuple(int(g) for g in plots.grid_size),
            snapshot_steps=tuple(int(t) for t in plots.snapshot_steps),
        ),
    )


def run_evaluation(cfg: DictConfig) -> dict[str, Any]:
    """Roll out a trained checkpoint (or a zero-parameter baseline) on a split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(cfg.seed))
    if device.type == "cuda":
        torch.set_float32_matmul_precision(str(cfg.precision.matmul_precision))
    split = str(cfg.rollout.split)
    output_root = Path(str(cfg.output_root)).expanduser()

    if cfg.get("checkpoint"):
        ckpt_path = resolve_checkpoint(str(cfg.checkpoint), output_root)
        ckpt = load_checkpoint(ckpt_path)
        train_cfg = OmegaConf.create(ckpt["config"])
        model_cfg = train_cfg.model
        if cfg.get("model") is not None:
            model_cfg = OmegaConf.merge(model_cfg, cfg.model)
        # Field and parameter settings must match training; the data location and
        # subset may differ (e.g. evaluate a G1 model on the full test set). The field
        # normalizer is restored from the checkpoint, so no stats.yaml is required.
        location = {k: cfg.data[k] for k in ("root", "source", "hf_repo", "subset", "limit")}
        if cfg.data.get("stats_path") is not None:
            location["stats_path"] = cfg.data.stats_path
        data_cfg = OmegaConf.merge(train_cfg.data, location)
        normalizer_state = ckpt.get("normalizer")
        if normalizer_state is not None:
            data_cfg.normalization = "none"
        context_frames = int(train_cfg.task.context_frames)
        predict_residual = bool(train_cfg.task.predict_residual)
        residual_scale = ckpt.get("residual_scale")
        model = build_model(model_cfg)
        load_model_weights(model, ckpt["model"])
        label = f"{model_cfg.name} @ {ckpt_path} (step {ckpt.get('step', '?')})"
        default_out = ckpt_path.parent / "eval" / split
    else:
        if cfg.get("model") is None:
            raise ValueError(
                "Pass checkpoint=<path|run> for a trained model or model=<name> for a baseline"
            )
        model_cfg = cfg.model
        data_cfg = cfg.data
        context_frames = int(cfg.task.context_frames)
        predict_residual = bool(cfg.task.predict_residual)
        residual_scale = None
        model = build_model(model_cfg)
        label = f"{model_cfg.name} (baseline)"
        default_out = output_root / "runs" / f"{model_cfg.name}-baseline" / "eval" / split

    output_dir = Path(str(cfg.output_dir)).expanduser() if cfg.get("output_dir") else default_out
    data = TurtlesData(data_cfg, splits=(split,))
    if cfg.get("checkpoint") and normalizer_state is not None:
        data.field_normalizer = FieldNormalizer.from_state_dict(normalizer_state)
    settings = _rollout_settings(
        cfg, context_frames=context_frames, predict_residual=predict_residual
    )
    settings.residual_scale = None if residual_scale is None else [float(v) for v in residual_scale]
    logger.info("Evaluating %s on %s | %s", label, split, data.summary())
    evaluator = RolloutEvaluator(model, data, settings, output_dir=output_dir, device=device)
    report = evaluator.evaluate()
    for key in (
        "rel_l2/0_1",
        "vrmse/0_1",
        f"vrmse/0_{max(settings.cumulative_windows)}",
        "diverged",
        "time/gpu_seconds",
    ):
        if key in report["aggregate"]:
            logger.info("  %-22s %.5g", key, report["aggregate"][key])
    return report
