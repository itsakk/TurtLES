"""Fixtures: a tiny synthetic dataset in the exact on-disk layout of TurtLES."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from turtles.data.io import FIELDS

# Structured mini-mesh: NX x NY (x, y) locations, each carrying a ring of LZ nodes along z.
NX, NY, LZ = 4, 3, 8
NUM_POINTS = NX * NY * LZ

CASES = {
    "train": ["Re1000_alpha0_a0p5_n2", "Re2000_alpha10_a1p0_ninf"],
    "val": ["Re1500_alpha5_a0p3_n1"],
    "test": ["Re3000_alpha30_a0p7_n5"],
    "reference": ["Re1000_alpha0_a0p5_n2"],
}


def make_coords(seed: int = 0) -> np.ndarray:
    """Shuffled structured coordinates ``(NUM_POINTS, 3)`` with periodic z rings."""
    xs = np.linspace(-1.0, 2.0, NX)
    ys = np.linspace(-0.5, 0.5, NY)
    zs = np.arange(LZ) * (2 * math.pi / LZ)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    perm = np.random.default_rng(seed).permutation(grid.shape[0])
    return grid[perm].astype(np.float32)


def make_frame(coords: np.ndarray, t: int, seed: int) -> np.ndarray:
    """Smooth, time-dependent fields plus noise, ``(N, 5)`` in ``p, rho, u, v, w`` order."""
    rng = np.random.default_rng(seed * 1000 + t)
    x, y, z = coords.T
    phase = 0.05 * t
    u = 0.2 + 0.1 * np.cos(2 * z + phase) * np.exp(-(y**2)) + 0.01 * rng.standard_normal(len(x))
    v = 0.05 * np.sin(x + phase) * np.cos(z) + 0.005 * rng.standard_normal(len(x))
    w = 0.02 * np.sin(3 * z + phase) + 0.005 * rng.standard_normal(len(x))
    p = 1.0 - 0.02 * u + 0.001 * rng.standard_normal(len(x))
    rho = 1.0 - 0.015 * u + 0.001 * rng.standard_normal(len(x))
    return np.stack([p, rho, u, v, w], axis=-1).astype(np.float32)


def write_case(case_dir: Path, *, num_frames: int, seed: int) -> None:
    """Write ``<case>_coords.h5`` and the 100-frame chunk files."""
    case_dir.mkdir(parents=True, exist_ok=True)
    name = case_dir.name
    coords = make_coords(seed)
    with h5py.File(case_dir / f"{name}_coords.h5", "w") as handle:
        for axis, column in zip(("x", "y", "z"), range(3), strict=True):
            handle.create_dataset(axis, data=coords[:, column])
    for start in range(0, num_frames, 100):
        with h5py.File(case_dir / f"{name}_b{start}_{start + 99}.h5", "w") as handle:
            for t in range(start, min(start + 100, num_frames)):
                group = handle.create_group(f"sol_t{t}")
                frame = make_frame(coords, t, seed)
                for c, field in enumerate(FIELDS):
                    group.create_dataset(field, data=frame[:, c])


def write_stats(path: Path) -> None:
    """Approximate statistics matching :func:`make_frame`."""
    stats = {
        "mean": {"p": 0.996, "rho": 0.997, "u": 0.2, "v": 0.0, "w": 0.0},
        "std": {"p": 0.002, "rho": 0.002, "u": 0.07, "v": 0.03, "w": 0.015},
        "rms": {"p": 0.996, "rho": 0.997, "u": 0.21, "v": 0.03, "w": 0.015},
        "mean_delta": dict.fromkeys(FIELDS, 0.0),
        "rms_delta": {"p": 0.0015, "rho": 0.0015, "u": 0.02, "v": 0.01, "w": 0.01},
        "std_delta": {"p": 0.0015, "rho": 0.0015, "u": 0.02, "v": 0.01, "w": 0.01},
    }
    path.write_text(yaml.safe_dump(stats))


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Dataset root mirroring the hub layout (``root/data/...``) with 200-frame cases."""
    root = tmp_path_factory.mktemp("turtles_root")
    data = root / "data"
    seed = 0
    for split, names in CASES.items():
        for name in names:
            frames = 100 if split == "reference" else 200
            write_case(data / split / name, num_frames=frames, seed=seed)
            seed += 1
    write_stats(data / "stats.yaml")
    return root


@pytest.fixture
def data_cfg(synthetic_root: Path) -> dict:
    """``data`` config section pointing at the synthetic root."""
    return {
        "root": str(synthetic_root),
        "source": "local",
        "subset": "full",
        "fields": list(FIELDS),
        "params": ["Re"],
        "normalize_params": True,
        "normalization": "zscore",
        "num_points": 64,
        "sampling": "contiguous",
        "window_stride": 1,
        "limit": {"train": None, "val": None, "test": None},
    }
