from __future__ import annotations

import math

import pytest
import torch

from turtles.data.dataset import TurtlesData, build_loader
from turtles.models import build_model
from turtles.training.checkpoint import load_checkpoint, resolve_checkpoint, strip_prefixes
from turtles.training.distributed import DistInfo
from turtles.training.schedule import build_lr_scheduler, build_optimizer, lr_multiplier
from turtles.training.trainer import Trainer, TrainerConfig

TINY = {
    "name": "calm_pde",
    "space_dim": 3,
    "fun_dim": 5,
    "out_dim": 5,
    "num_steps": 1,
    "num_params": 1,
    "num_latent_points": 8,
    "knn_k": 4,
    "kernel_hidden": 8,
    "dim": 24,
    "num_heads": 4,
    "depth": 1,
    "coord_min": None,
    "coord_max": None,
    "use_checkpoint": False,
}


def make_trainer(tmp_path, data_cfg, steps=4, train_query_points=16, **overrides):
    data = TurtlesData(data_cfg, splits=("train", "val"))
    window = 2
    train = build_loader(
        data.window_dataset("train", window=window), batch_size=2, shuffle=True, num_workers=0
    )
    val = build_loader(
        data.window_dataset("val", window=window), batch_size=2, shuffle=False, num_workers=0
    )
    model = build_model(TINY)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    scheduler = build_lr_scheduler(optimizer, total_steps=steps, warmup_steps=1)
    cfg = TrainerConfig(
        steps=steps,
        amp=False,
        log_every=2,
        save_every=2,
        val_every=2,
        val_max_batches=1,
        train_query_points=train_query_points,
        **overrides,
    )
    trainer = Trainer(
        model,
        train_loader=train,
        val_loader=val,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg,
        run_dir=tmp_path,
        device=torch.device("cpu"),
        dist=DistInfo(0, 0, 1),
        channel_names=data.fields,
        checkpoint_extra={"config": {"x": 1}},
    )
    return trainer, train


def test_lr_schedule_shape():
    kw = dict(total_steps=100, warmup_steps=10, schedule="cosine", min_ratio=0.01)
    assert lr_multiplier(0, **kw) == pytest.approx(0.1)
    assert lr_multiplier(9, **kw) == pytest.approx(1.0)
    assert lr_multiplier(10, **kw) == pytest.approx(1.0)
    assert lr_multiplier(100, **kw) == pytest.approx(0.01)
    assert lr_multiplier(55, **kw) == pytest.approx(0.505, abs=1e-6)
    assert lr_multiplier(50, **{**kw, "schedule": "constant"}) == 1.0
    with pytest.raises(ValueError):
        lr_multiplier(50, **{**kw, "schedule": "linear"})


def test_compute_loss_and_metrics(tmp_path, data_cfg):
    trainer, loader = make_trainer(tmp_path, data_cfg)
    loss, metrics = trainer.compute_loss(next(iter(loader)))
    assert loss.dim() == 0 and math.isfinite(loss.item())
    assert "rmse" in metrics and "rmse_channel/u" in metrics
    assert "loss_persistence" in metrics and "skill" in metrics
    assert "pred_scale" in metrics and "skill_channel/w" in metrics
    expected_skill = 1 - loss.item() / metrics["loss_persistence"].item()
    assert metrics["skill"].item() == pytest.approx(expected_skill, abs=1e-6)  # skill ~ 0 untrained


def test_fit_writes_checkpoints_and_resumes(tmp_path, data_cfg):
    trainer, _ = make_trainer(tmp_path, data_cfg, steps=4)
    best = trainer.fit()
    assert math.isfinite(best) and trainer.step == 4
    assert (tmp_path / "last.pt").exists() and (tmp_path / "best.pt").exists()
    ckpt = load_checkpoint(tmp_path / "last.pt")
    assert ckpt["step"] == 4 and ckpt["config"] == {"x": 1}

    resumed, _ = make_trainer(tmp_path / "resumed", data_cfg, steps=6)
    resumed.load(tmp_path / "last.pt")
    assert resumed.step == 4
    for a, b in zip(
        trainer.raw_model.state_dict().values(),
        resumed.raw_model.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(a, b)
    resumed.fit()
    assert resumed.step == 6


def test_loss_decreases_on_fixed_batch(tmp_path, data_cfg):
    trainer, loader = make_trainer(tmp_path, data_cfg, steps=30, train_query_points=None)
    batch = next(iter(loader))
    with torch.no_grad():
        before, _ = trainer.compute_loss(batch)
    trainer.cfg.save_every = 0
    trainer.cfg.val_every = 0
    trainer.val_loader = None
    for _ in range(30):
        trainer.step += 1
        trainer._train_step(batch, boundary=True)
    with torch.no_grad():
        after, _ = trainer.compute_loss(batch)
    assert after.item() < before.item()


def test_strip_prefixes_and_resolve(tmp_path):
    state = {"_orig_mod.module.w": torch.ones(1), "b": torch.zeros(1)}
    assert set(strip_prefixes(state)) == {"w", "b"}
    run_dir = tmp_path / "runs" / "myrun"
    run_dir.mkdir(parents=True)
    (run_dir / "last.pt").write_bytes(b"x")
    assert resolve_checkpoint("myrun", tmp_path) == run_dir / "last.pt"
    (run_dir / "best.pt").write_bytes(b"x")
    assert resolve_checkpoint(run_dir, None) == run_dir / "best.pt"
    assert resolve_checkpoint(run_dir / "last.pt", None) == run_dir / "last.pt"
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("missing", tmp_path)


def test_residual_scale_balances_channels(tmp_path, data_cfg):
    data = TurtlesData(data_cfg, splits=("train",))
    scale = data.increment_scale().tolist()
    trainer, loader = make_trainer(
        tmp_path, data_cfg, residual_scale=scale, train_query_points=None
    )
    _, metrics = trainer.compute_loss(next(iter(loader)))
    # With the scaled target, the persistence loss is O(1) per channel instead of std_delta^2/std^2.
    assert 0.1 < metrics["loss_persistence"].item() < 10
    assert math.isfinite(metrics["rmse"].item())
