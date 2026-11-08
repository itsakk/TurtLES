from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from tests.conftest import LZ, NX, NY, make_coords
from turtles.data.dataset import TurtlesData
from turtles.evaluation.metrics import (
    RolloutRecord,
    aggregate_summaries,
    frame_errors,
    summarize_record,
)
from turtles.evaluation.predict import predict_frame, tile_partition_predict
from turtles.evaluation.rollout import PlotSettings, RolloutEvaluator, RolloutSettings
from turtles.evaluation.slices import apply_grid_projection, build_grid_projection
from turtles.evaluation.spectra import (
    band_log_rmse,
    band_masks,
    build_spanwise_rings,
    spanwise_power,
)
from turtles.models import build_model


def test_frame_errors_match_formulas():
    torch.manual_seed(0)
    target = torch.randn(100, 3) + torch.tensor([5.0, 0.0, -2.0])
    pred = target + 0.1 * torch.randn(100, 3)
    errors = frame_errors(pred, target, eps=0.0)
    t, p = target.double().numpy(), pred.double().numpy()
    err = np.linalg.norm(p - t, axis=0)
    np.testing.assert_allclose(errors.rel_l2.numpy(), err / np.linalg.norm(t, axis=0), rtol=1e-6)
    np.testing.assert_allclose(
        errors.vrmse.numpy(), err / np.linalg.norm(t - t.mean(0), axis=0), rtol=1e-6
    )
    # The per-channel spatial mean is the VRMSE = 1 predictor.
    mean_pred = target.mean(dim=0, keepdim=True).expand_as(target)
    assert frame_errors(mean_pred, target).vrmse.numpy() == pytest.approx(np.ones(3), rel=1e-6)


def test_record_and_windows():
    record = RolloutRecord(["a", "b"], ["low", "high"])
    for t in range(10):
        err = frame_errors(
            torch.full((4, 2), float(t + 1)),
            torch.ones(4, 2) * torch.tensor([1.0, 2.0]) + torch.arange(4).unsqueeze(1),
        )
        record.add(err, spectral=[0.1 * t, 0.2], predict_seconds=0.5)
    summary = summarize_record(
        record,
        cumulative_windows=[1, 5, 10, 50],
        disjoint_windows=[(1, 5), (6, 10)],
        divergence_threshold=1e9,
        solver_gpu_seconds=100.0,
    )
    arrays = record.arrays()
    assert summary["rel_l2/0_1"] == pytest.approx(arrays["rel_l2"][0].mean())
    assert summary["vrmse/0_5"] == pytest.approx(arrays["vrmse"][:5].mean(axis=1).mean())
    assert summary["vrmse/win_6_10"] == pytest.approx(arrays["vrmse"][5:10].mean(axis=1).mean())
    assert summary["spec_low/0_10"] == pytest.approx(np.mean([0.1 * t for t in range(10)]))
    assert summary["vrmse/0_50"] == summary["vrmse/0_10"]  # clipped to the available steps
    assert summary["diverged"] == 0.0
    assert summary["time/gpu_seconds"] == pytest.approx(5.0)
    assert summary["time/speedup_vs_solver"] == pytest.approx(20.0)
    assert summary["vrmse_channel/a"] == pytest.approx(arrays["vrmse"][:, 0].mean())

    record.add(None, finite=False)
    assert summarize_record(record, cumulative_windows=[11])["diverged"] == 1.0
    agg = aggregate_summaries([{"x": 1.0, "diverged": 1.0}, {"x": 3.0, "diverged": 0.0}])
    assert agg["x"] == 2.0 and agg["diverged"] == 0.5 and agg["num_trajectories"] == 2


def test_spanwise_rings_and_spectrum():
    coords = torch.from_numpy(make_coords(seed=3))
    rings = build_spanwise_rings(coords)
    assert rings.num_rings == NX * NY and rings.length == LZ
    assert rings.wavenumbers.tolist() == list(range(LZ // 2 + 1))
    z = coords[:, 2]
    # Dominant k = 3 mode plus weaker modes so every band wavenumber carries power.
    field = (
        torch.cos(3 * z) + 0.3 * torch.cos(z) + 0.2 * torch.cos(2 * z) + 0.2 * torch.cos(4 * z)
    ).unsqueeze(-1)
    power = spanwise_power(field, rings)
    assert power.shape == (1, LZ // 2 + 1)
    assert int(power[0].argmax()) == 3
    assert power[0, 3] > 5 * power[0, 1]
    masks = band_masks(rings.wavenumbers, {"low": (1, 2), "high": (3, 4)})
    assert masks["low"].sum() == 2 and masks["high"].sum() == 2
    same = band_log_rmse(power[0], power[0], masks)
    assert same["low"] == 0.0 and same["high"] == 0.0
    off = band_log_rmse(power[0] * math.e, power[0], masks)
    assert off["high"] == pytest.approx(1.0, rel=1e-6)


def test_grid_projection():
    coords = torch.from_numpy(make_coords(seed=1))
    proj = build_grid_projection(coords, grid_size=(8, 6), knn_k=3)
    field = coords[:, :1].clone()  # x itself: projection should recover x along columns
    grid = apply_grid_projection(field, proj)
    assert grid.shape == (8, 6, 1) and torch.isfinite(grid).all()
    assert grid[0, -1, 0] > grid[0, 0, 0]


class _PerNode(torch.nn.Module):
    def forward(self, context, coords, params=None, query_coords=None):
        return context[:, -1] * 2 + coords[..., :1]


def test_tile_partition_equals_full():
    model = _PerNode()
    context, coords = torch.randn(1, 1, 50, 5), torch.randn(1, 50, 3)
    full = model(context, coords)
    tiled = tile_partition_predict(model, context, coords, None, chunk_size=7)
    torch.testing.assert_close(full, tiled)
    torch.testing.assert_close(
        predict_frame(model, context, coords, None, query_chunk_size=7), full
    )


def test_rollout_evaluator_end_to_end(tmp_path, data_cfg):
    data = TurtlesData({**data_cfg, "params": []}, splits=("test",))
    model = build_model({"name": "persistence", "fun_dim": 5, "out_dim": 5})
    settings = RolloutSettings(
        split="test",
        steps=12,
        query_chunk_size=40,
        predict_residual=False,
        amp=False,
        cumulative_windows=(1, 5, 12),
        disjoint_windows=((1, 6), (7, 12)),
        loader_workers=0,
        spectral_bands={"low": (1, 2), "high": (3, 4)},
        solver_gpu_seconds=10.0,
        plots=PlotSettings(
            enabled=True, max_trajectories=1, grid_size=(8, 6), snapshot_steps=(0, 5, 11)
        ),
    )
    out = tmp_path / "eval"
    report = RolloutEvaluator(
        model, data, settings, output_dir=out, device=torch.device("cpu")
    ).evaluate()
    agg = report["aggregate"]
    assert agg["num_trajectories"] == 1 and agg["diverged"] == 0.0
    assert 0 < agg["rel_l2/0_1"] < 1 and agg["vrmse/win_7_12"] > agg["vrmse/0_1"]
    assert "spec_low/0_12" in agg and math.isfinite(agg["spec_high/0_1"])
    assert (out / "metrics.json").exists() and (out / "summary.md").exists()
    case_dir = out / "trajectories" / "Re3000_alpha30_a0p7_n5"
    for name in (
        "per_frame.npz",
        "spectra.npz",
        "summary.json",
        "error_curves.pdf",
        "snapshots_u.pdf",
        "spectra.pdf",
    ):
        assert (case_dir / name).exists(), name
    per_frame = np.load(case_dir / "per_frame.npz")
    assert per_frame["rel_l2"].shape == (12, 5)
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["settings"]["split"] == "test"
