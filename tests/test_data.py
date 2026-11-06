from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import CASES, LZ, NUM_POINTS, make_frame
from turtles.data.dataset import (
    FrameDataset,
    TurtlesData,
    WindowDataset,
    build_loader,
    select_nodes,
)
from turtles.data.hub import is_available, resolve_data_dir
from turtles.data.io import FIELDS, Trajectory, discover_trajectories, read_coords, read_frame
from turtles.data.normalization import FieldNormalizer, load_stats
from turtles.data.stats import compute_stats
from turtles.data.subsets import parse_subset


def test_resolve_data_dir_handles_hub_layout(synthetic_root):
    assert resolve_data_dir(synthetic_root) == synthetic_root / "data"
    assert resolve_data_dir(synthetic_root / "data") == synthetic_root / "data"


def test_trajectory_discovery_and_frames(synthetic_root):
    data = synthetic_root / "data"
    train = discover_trajectories(data / "train")
    assert [t.name for t in train] == CASES["train"]
    traj = train[0]
    assert traj.num_frames == 200 and len(traj.chunk_paths) == 2
    assert traj.num_points == NUM_POINTS
    assert traj.params.Re == 1000
    capped = Trajectory(traj.directory, max_frames=100)
    assert capped.num_frames == 100
    with pytest.raises(IndexError):
        capped.chunk_path(150)


def test_read_frame_matches_generator(synthetic_root):
    traj = discover_trajectories(synthetic_root / "data" / "train")[0]
    coords = read_coords(traj)
    frame = read_frame(traj, 7, FIELDS)
    expected = make_frame(coords, 7, seed=0)
    np.testing.assert_allclose(frame, expected, rtol=1e-6)
    sel = slice(10, 30)
    np.testing.assert_allclose(
        read_frame(traj, 7, ("u", "p"), sel), expected[10:30][:, [2, 0]], rtol=1e-6
    )
    idx = np.array([3, 5, 40])
    np.testing.assert_allclose(read_coords(traj, idx), coords[idx])


def test_subset_filtering_and_limit(synthetic_root):
    data = synthetic_root / "data"
    g1 = discover_trajectories(data / "train", subset=parse_subset("G1-T100"), max_frames=100)
    assert [t.name for t in g1] == ["Re1000_alpha0_a0p5_n2"] and g1[0].num_frames == 100
    assert discover_trajectories(data / "train", limit=1)[0].name == CASES["train"][0]
    assert discover_trajectories(data / "missing") == []


def test_select_nodes_modes():
    np.random.seed(0)
    assert select_nodes(10, None, "contiguous") is None
    assert select_nodes(10, 20, "contiguous") is None
    window = select_nodes(100, 10, "contiguous")
    assert isinstance(window, slice) and window.stop - window.start == 10
    idx = select_nodes(100, 10, "random")
    assert len(idx) == 10 and np.all(np.diff(idx) > 0)
    with pytest.raises(ValueError):
        select_nodes(100, 10, "weird")


def test_window_dataset_shapes(synthetic_root):
    trajectories = discover_trajectories(synthetic_root / "data" / "train")
    ds = WindowDataset(trajectories, window=2, num_points=32, stride=50)
    assert len(ds) == 2 * len(range(0, 199, 50))
    sample = ds[0]
    assert sample["fields"].shape == (2, 32, 5)
    assert sample["coords"].shape == (32, 3)
    assert sample["params"].shape == (0,)
    loader = build_loader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["fields"].shape == (2, 2, 32, 5) and batch["params"].shape == (2, 0)


def test_frame_dataset(synthetic_root):
    traj = discover_trajectories(synthetic_root / "data" / "test")[0]
    ds = FrameDataset(traj, start=5, end=9)
    assert len(ds) == 4 and ds[0].shape == (NUM_POINTS, 5) and ds.coords().shape == (NUM_POINTS, 3)
    sub = FrameDataset(traj, node_indices=np.arange(0, 20))
    assert sub[0].shape == (20, 5) and sub.num_points == 20


def test_field_normalizer_roundtrip(synthetic_root):
    stats = load_stats(synthetic_root / "data" / "stats.yaml")
    for strategy in ("zscore", "zscore_delta", "rms"):
        norm = FieldNormalizer.from_stats(stats, FIELDS, strategy)
        x = torch.randn(7, 5) * 0.1 + 0.5
        torch.testing.assert_close(norm.unnormalize(norm.normalize(x)), x)
        restored = FieldNormalizer.from_state_dict(norm.state_dict())
        np.testing.assert_allclose(restored.scale, norm.scale)
    identity = FieldNormalizer.from_stats(stats, FIELDS, "none")
    assert not identity.enabled
    with pytest.raises(ValueError):
        FieldNormalizer.from_stats(stats, FIELDS, "minmax")


def test_turtles_data_end_to_end(data_cfg):
    data = TurtlesData(data_cfg, splits=("train", "val", "test"))
    assert data.field_normalizer.enabled and data.num_channels == 5
    assert len(data.trajectories("train")) == 2 and len(data.trajectories("test")) == 1
    ds = data.window_dataset("train", window=2)
    sample = ds[3]
    assert sample["fields"].shape == (2, 64, 5) and sample["params"].shape == (1,)
    # Normalized values are O(1).
    assert sample["fields"].abs().mean() < 10
    params = data.params_tensor(data.trajectories("train")[0])
    assert params.shape == (1, 1)
    assert data.check_batchable("train", batch_size=2) == 64
    big = TurtlesData({**data_cfg, "num_points": NUM_POINTS + 1})
    assert big.check_batchable("train", 2) == NUM_POINTS  # clamped to the smallest mesh
    assert big.window_dataset("train", window=2)[0]["fields"].shape[1] == NUM_POINTS


def test_turtles_data_debug_subset_and_limit(data_cfg):
    debug = TurtlesData({**data_cfg, "subset": "debug"}, splits=("train",))
    assert [t.name for t in debug.trajectories("train")] == ["Re1000_alpha0_a0p5_n2"]
    assert debug.trajectories("train")[0].num_frames == 100
    limited = TurtlesData({**data_cfg, "limit": {"train": 1}})
    assert len(limited.trajectories("train")) == 1


def test_turtles_data_missing_root():
    with pytest.raises(FileNotFoundError):
        TurtlesData({"root": "/nonexistent/turtles", "source": "local"})


def test_is_available(synthetic_root):
    data = synthetic_root / "data"
    assert is_available(data, parse_subset("G4-T200"), ("train", "val", "test"))
    assert not is_available(data, parse_subset("full"), ("train",))  # needs 400 frames
    assert is_available(data, parse_subset("debug"), ("train",))


def test_compute_stats_matches_numpy(synthetic_root):
    traj = discover_trajectories(synthetic_root / "data" / "train", limit=1)
    stats = compute_stats(traj, FIELDS, frame_stride=50, max_nodes=None)
    frames = np.stack([read_frame(traj[0], t) for t in range(0, 199, 50)])
    assert stats["mean"]["u"] == pytest.approx(float(frames[..., 2].mean()), rel=1e-3)
    assert stats["std"]["v"] == pytest.approx(float(frames[..., 3].std()), rel=1e-2)
    assert set(stats) == {"mean", "std", "rms", "mean_delta", "rms_delta", "std_delta"}
    assert LZ == 8  # sanity: ring length used by the spectral tests


def test_strided_selection_and_increment_scale(data_cfg):
    np.random.seed(1)
    sel = select_nodes(1000, 100, "strided")
    idx = np.concatenate([np.arange(1000)[part] for part in sel])
    assert len(idx) == 100 and np.diff(idx).min() >= 9 and np.diff(idx).max() <= 11
    data = TurtlesData({**data_cfg, "sampling": "strided", "num_points": 48})
    sample = data.window_dataset("train", window=2)[0]
    assert sample["fields"].shape == (2, 48, 5)
    scale = data.increment_scale()
    assert scale.shape == (5,) and (scale > 0).all()
    assert scale[0].item() == pytest.approx(0.0015 / 0.002, rel=1e-5)  # std_delta / std of p


@pytest.mark.parametrize(
    "total,num_points", [(1000, 100), (1500, 1000), (5_237_848, 2_933_282), (10, 9), (37, 5)]
)
def test_strided_slices_cover_the_whole_range(total, num_points):
    np.random.seed(0)
    for jitter in (True, False):
        sel = select_nodes(total, num_points, "strided", jitter=jitter)
        idx = np.concatenate([np.arange(total)[part] for part in sel])
        assert len(idx) == num_points
        assert np.all(np.diff(idx) > 0) and idx[0] >= 0 and idx[-1] < total
        gaps = np.diff(np.concatenate([[-1], idx, [total]]))
        assert gaps.max() <= total // num_points + 2  # no unread region
    fixed = [select_nodes(total, num_points, "strided", jitter=False) for _ in range(2)]
    assert fixed[0] == fixed[1]


def test_strided_reads_match_index_reads(synthetic_root):
    from turtles.data.io import discover_trajectories, read_coords

    traj = discover_trajectories(synthetic_root / "data" / "train")[0]
    sel = select_nodes(traj.num_points, traj.num_points // 3, "strided", jitter=False)
    idx = np.concatenate([np.arange(traj.num_points)[part] for part in sel])
    np.testing.assert_array_equal(read_coords(traj, sel), read_coords(traj)[idx])
