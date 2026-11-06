from __future__ import annotations

import json

import pytest
import torch

from turtles.experiment import load_config, run_evaluation, run_training


def test_load_config_composes_presets():
    cfg = load_config("train", ["experiment=paper/gino", "data.subset=G1-T100"])
    assert (
        cfg.model.name == "gino" and cfg.optim.accumulate_gradients == 2 and not cfg.precision.amp
    )
    assert cfg.data.subset == "G1-T100" and cfg.model.fun_dim == 5 and cfg.model.num_params == 1
    cfg = load_config("eval", ["model=mean_field"])
    assert cfg.model.name == "mean_field" and cfg.task.predict_residual is False


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_train_then_evaluate_end_to_end(tmp_path, synthetic_root):
    overrides = [
        "experiment=debug",
        f"data.root={synthetic_root}",
        "data.subset=full",
        "data.num_points=64",
        "model.num_latent_points=8",
        "model.knn_k=4",
        "model.kernel_hidden=8",
        "model.dim=24",
        "model.num_heads=4",
        "model.coord_min=null",
        "model.coord_max=null",
        "optim.steps=3",
        "optim.batch_size=2",
        "loader.num_workers=0",
        "validation.every=2",
        "checkpoint.save_every=2",
        "precision.amp=false",
        f"output_root={tmp_path}",
        "run_name=smoke",
    ]
    cfg = load_config("train", overrides)
    best = run_training(cfg)
    run_dir = tmp_path / "runs" / "smoke"
    assert torch.isfinite(torch.tensor(best))
    assert (
        (run_dir / "last.pt").exists()
        and (run_dir / "best.pt").exists()
        and (run_dir / "config.yaml").exists()
    )

    eval_cfg = load_config(
        "eval",
        [
            f"checkpoint={run_dir}",
            f"data.root={synthetic_root}",
            "rollout.steps=4",
            "rollout.query_chunk_size=50",
            "metrics.cumulative_windows=[1,4]",
            "metrics.disjoint_windows=[[1,2],[3,4]]",
            "metrics.spectral.bands.low=[1,2]",
            "metrics.spectral.bands.mid=[2,3]",
            "metrics.spectral.bands.high=[3,4]",
            "plots.max_trajectories=1",
            "plots.grid_size=[8,6]",
            "plots.snapshot_steps=[0,3]",
            "loader.num_workers=0",
            "precision.amp=false",
            f"output_root={tmp_path}",
        ],
    )
    report = run_evaluation(eval_cfg)
    assert report["aggregate"]["num_trajectories"] == 1
    # Evaluation restores the normalizer from the checkpoint: no stats.yaml needed.
    stats = synthetic_root / "data" / "stats.yaml"
    backup = stats.read_text()
    stats.unlink()
    try:
        no_stats_cfg = load_config(
            "eval",
            [
                f"checkpoint={run_dir}",
                f"data.root={synthetic_root}",
                "rollout.steps=2",
                "rollout.query_chunk_size=50",
                "metrics.cumulative_windows=[1,2]",
                "metrics.disjoint_windows=[[1,2]]",
                "metrics.spectral.bands.low=[1,2]",
                "metrics.spectral.bands.mid=[2,3]",
                "metrics.spectral.bands.high=[3,4]",
                "plots.enabled=false",
                "loader.num_workers=0",
                "precision.amp=false",
                f"output_root={tmp_path}",
                f"output_dir={tmp_path / 'nostats'}",
            ],
        )
        again = run_evaluation(no_stats_cfg)
        assert again["aggregate"]["rel_l2/0_1"] == pytest.approx(
            report["aggregate"]["rel_l2/0_1"], rel=1e-6
        )
    finally:
        stats.write_text(backup)
    assert (run_dir / "eval" / "test" / "metrics.json").exists()
    saved = json.loads((run_dir / "eval" / "test" / "metrics.json").read_text())
    assert "rel_l2/0_4" in saved["aggregate"]

    # Baseline path: no checkpoint, model selected on the command line.
    base_cfg = load_config(
        "eval",
        [
            "model=persistence",
            f"data.root={synthetic_root}",
            "data.params=[]",
            "rollout.steps=3",
            "rollout.query_chunk_size=50",
            "metrics.cumulative_windows=[1,3]",
            "metrics.disjoint_windows=[[1,3]]",
            "metrics.spectral.bands.low=[1,2]",
            "metrics.spectral.bands.mid=[2,3]",
            "metrics.spectral.bands.high=[3,4]",
            "plots.enabled=false",
            "loader.num_workers=0",
            "precision.amp=false",
            f"output_root={tmp_path}",
        ],
    )
    base = run_evaluation(base_cfg)
    assert (tmp_path / "runs" / "persistence-baseline" / "eval" / "test" / "metrics.json").exists()
    assert base["aggregate"]["diverged"] == 0.0
