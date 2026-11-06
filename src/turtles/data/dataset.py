"""PyTorch datasets over TurtLES trajectories.

* :class:`WindowDataset` yields windows of consecutive frames on a random node
  subset (training and validation).
* :class:`FrameDataset` streams the frames of a single trajectory one at a time
  (full-mesh autoregressive evaluation; a frame of the largest mesh is ~200 MB).
* :class:`TurtlesData` turns the ``data`` section of a Hydra config into the
  trajectories, normalizers and datasets of each split.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from turtles.data.hub import ensure_available, resolve_data_dir
from turtles.data.io import (
    FIELDS,
    Trajectory,
    discover_trajectories,
    read_coords,
    read_frame,
    read_frames,
)
from turtles.data.normalization import STATS_FILENAME, FieldNormalizer, ParamNormalizer
from turtles.data.subsets import SPLITS, Subset, parse_subset

logger = logging.getLogger(__name__)

SAMPLING_MODES: tuple[str, ...] = ("contiguous", "strided", "random")


def seed_worker(worker_id: int) -> None:  # signature imposed by DataLoader
    """Seed NumPy in each DataLoader worker from the torch per-worker seed."""
    np.random.seed(torch.initial_seed() % 2**32)


def strided_slices(
    total: int, num_points: int, *, blocks: int = 8, jitter: bool = True
) -> tuple[slice, ...]:
    """``num_points`` nodes spread over all of ``0 .. total-1`` as a few strided slices.

    The node range is cut into ``blocks`` equal parts. Within a part of ``n`` nodes that
    must give ``p`` points, the first ``r = n - s * p`` points are read every ``s + 1``
    nodes and the rest every ``s = n // p``, which spans the part exactly. A single stride
    ``total // num_points`` would leave the last ``total - stride * num_points`` nodes
    unread: up to half of a mesh, and a contiguous spatial region since nodes are stored
    element by element. ``jitter`` draws the common start offset in ``[0, min stride)``;
    without it the selection is a fixed function of ``total`` (a trajectory always gets
    the same nodes, which lets mesh-dependent caches hit).
    """
    if not 0 < num_points < total:
        raise ValueError(f"need 0 < num_points < total, got {num_points} and {total}")
    nodes = [j * total // blocks for j in range(blocks + 1)]
    points = [j * num_points // blocks for j in range(blocks + 1)]
    parts = list(zip(nodes[:-1], nodes[1:], points[:-1], points[1:], strict=True))
    if any(p1 - p0 < 1 or n1 - n0 < p1 - p0 for n0, n1, p0, p1 in parts):
        parts = [(0, total, 0, num_points)]
    min_stride = min((n1 - n0) // (p1 - p0) for n0, n1, p0, p1 in parts)
    offset = int(np.random.randint(0, min_stride)) if jitter else 0
    slices: list[slice] = []
    for n0, n1, p0, p1 in parts:
        n, p = n1 - n0, p1 - p0
        s = n // p
        r = n - s * p
        start = n0 + offset
        mid = start + r * (s + 1)
        if r:
            slices.append(slice(start, mid, s + 1))
        slices.append(slice(mid, mid + (p - r) * s, s))
    return tuple(slices)


def select_nodes(
    total: int, num_points: int | None, sampling: str, *, jitter: bool = True
) -> slice | tuple[slice, ...] | np.ndarray | None:
    """Pick the nodes of one sample.

    ``strided`` spreads ``num_points`` nodes over the whole file, hence the whole domain,
    with a few strided hyperslab reads (:func:`strided_slices`; ``jitter`` randomises the
    offset). ``contiguous`` draws a random offset and reads ``num_points`` consecutive
    nodes: one sequential read, but nodes are stored element by element, so the window is
    a spatially localized sub-region. ``random`` draws nodes uniformly without replacement.
    ``None`` means every node (full mesh).
    """
    if num_points is None or num_points >= total:
        return None
    if sampling == "contiguous":
        offset = int(np.random.randint(0, total - num_points + 1))
        return slice(offset, offset + num_points)
    if sampling == "strided":
        return strided_slices(total, num_points, jitter=jitter)
    if sampling == "random":
        return np.sort(np.random.choice(total, num_points, replace=False))
    raise ValueError(f"Unknown sampling '{sampling}'; expected one of {SAMPLING_MODES}")


class WindowDataset(Dataset):
    """Sliding windows of ``window`` consecutive frames over a set of trajectories.

    Each item is a dict with
    ``fields`` ``(T, N, C)`` normalized fields, ``coords`` ``(N, 3)``,
    ``params`` ``(P,)`` normalized PDE parameters, plus the integer
    ``trajectory`` index and window ``start`` frame.
    """

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        *,
        window: int,
        num_points: int | None,
        fields: Sequence[str] = FIELDS,
        field_normalizer: FieldNormalizer | None = None,
        param_normalizer: ParamNormalizer | None = None,
        sampling: str = "contiguous",
        stride: int = 1,
    ):
        if window < 1:
            raise ValueError("window must be >= 1")
        if sampling not in SAMPLING_MODES:
            raise ValueError(f"Unknown sampling '{sampling}'; expected one of {SAMPLING_MODES}")
        self.trajectories = list(trajectories)
        self.window = int(window)
        self.num_points = None if num_points is None else int(num_points)
        self.fields = list(fields)
        self.field_normalizer = field_normalizer or FieldNormalizer.identity()
        self.param_normalizer = param_normalizer or ParamNormalizer(())
        self.sampling = sampling
        self._index: list[tuple[int, int]] = [
            (i, start)
            for i, traj in enumerate(self.trajectories)
            for start in range(0, traj.num_frames - self.window + 1, int(stride))
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        traj_idx, start = self._index[idx]
        traj = self.trajectories[traj_idx]
        sel = select_nodes(traj.num_points, self.num_points, self.sampling)
        frames = read_frames(traj, range(start, start + self.window), self.fields, sel)
        coords = read_coords(traj, sel)
        return {
            "fields": torch.from_numpy(self.field_normalizer.normalize(frames)),
            "coords": torch.from_numpy(coords),
            "params": torch.from_numpy(self.param_normalizer(traj.params)),
            "trajectory": traj_idx,
            "start": start,
        }


class FrameDataset(Dataset):
    """Frames ``start .. end-1`` of one trajectory, one normalized ``(N, C)`` tensor each.

    ``node_indices`` restricts every frame (and :meth:`coords`) to the same sorted
    node subset; ``None`` streams the full mesh.
    """

    def __init__(
        self,
        trajectory: Trajectory,
        *,
        fields: Sequence[str] = FIELDS,
        field_normalizer: FieldNormalizer | None = None,
        start: int = 0,
        end: int | None = None,
        node_indices: np.ndarray | None = None,
    ):
        self.trajectory = trajectory
        self.fields = list(fields)
        self.field_normalizer = field_normalizer or FieldNormalizer.identity()
        self.start = int(start)
        self.end = trajectory.num_frames if end is None else min(int(end), trajectory.num_frames)
        if self.end <= self.start:
            raise ValueError(f"empty frame range [{self.start}, {self.end})")
        self.node_indices = None if node_indices is None else np.asarray(node_indices)

    @property
    def num_points(self) -> int:
        """Nodes per frame after the optional subset."""
        return self.trajectory.num_points if self.node_indices is None else len(self.node_indices)

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = read_frame(self.trajectory, self.start + int(idx), self.fields, self.node_indices)
        return torch.from_numpy(self.field_normalizer.normalize(frame))

    def coords(self) -> torch.Tensor:
        """Node coordinates ``(N, 3)`` matching the frames."""
        return torch.from_numpy(read_coords(self.trajectory, self.node_indices))


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    prefetch_factor: int | None = 2,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool = False,
    distributed: bool = False,
    seed: int = 0,
    multiprocessing_context: Any = None,
) -> DataLoader:
    """DataLoader with the worker seeding and (optional) distributed sampler wired in."""
    sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed) if distributed else None
    workers = int(num_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        prefetch_factor=prefetch_factor if workers > 0 else None,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers and workers > 0,
        worker_init_fn=seed_worker if workers > 0 else None,
        multiprocessing_context=multiprocessing_context if workers > 0 else None,
    )


class TurtlesData:
    """Resolve the ``data`` config section into trajectories, normalizers and datasets.

    Args:
        cfg: Mapping with the keys of ``configs/data/turtles.yaml``.
        splits: Splits that must be present (downloaded first when ``source: hf``).
    """

    def __init__(self, cfg: Mapping[str, Any], *, splits: Sequence[str] = SPLITS):
        self.cfg = cfg
        self.subset: Subset = parse_subset(cfg.get("subset", "full"))
        self.fields = list(cfg.get("fields", FIELDS))
        self.param_names = list(cfg.get("params", ()) or ())
        self.num_points = cfg.get("num_points")
        self.sampling = str(cfg.get("sampling", "contiguous"))
        self.window_stride = int(cfg.get("window_stride", 1))
        limit = cfg.get("limit") or {}
        self.limit = {s: limit.get(s) for s in SPLITS}

        root = Path(str(cfg.get("root", "data/turtles"))).expanduser()
        if str(cfg.get("source", "local")).lower() == "hf":
            self.data_dir = ensure_available(
                root,
                self.subset,
                splits,
                repo_id=str(cfg.get("hf_repo", "turtlesprojectml/turtles")),
            )
        else:
            self.data_dir = resolve_data_dir(root)
            if not self.data_dir.exists():
                raise FileNotFoundError(
                    f"Data directory {self.data_dir} does not exist. Set data.root / "
                    "TURTLES_DATA_DIR, or use data.source=hf to download the subset."
                )

        stats_path = Path(str(cfg.get("stats_path") or self.data_dir / STATS_FILENAME)).expanduser()
        strategy = cfg.get("normalization", "zscore")
        if (strategy or "none") != "none" and not stats_path.exists():
            raise FileNotFoundError(
                f"{stats_path} not found; run `turtles-data stats`, point data.stats_path at a "
                "stats file, or set data.normalization=none"
            )
        self.field_normalizer = FieldNormalizer.from_file(stats_path, self.fields, strategy)
        self.param_normalizer = ParamNormalizer(
            self.param_names, enabled=bool(cfg.get("normalize_params", True))
        )
        self._trajectories: dict[str, list[Trajectory]] = {}

    @property
    def num_channels(self) -> int:
        """Number of field channels."""
        return len(self.fields)

    def split_dir(self, split: str) -> Path:
        """Directory of ``split`` for the configured subset."""
        return self.data_dir / self.subset.split_dir_name(split)

    def trajectories(self, split: str) -> list[Trajectory]:
        """Trajectories of ``split`` (cached), filtered and truncated by the subset."""
        if split not in self._trajectories:
            self._trajectories[split] = discover_trajectories(
                self.split_dir(split),
                subset=self.subset,
                max_frames=self.subset.num_frames,
                limit=self.limit.get(split),
            )
        return self._trajectories[split]

    def window_dataset(self, split: str, *, window: int) -> WindowDataset | None:
        """Window dataset for ``split`` or ``None`` when the split has no trajectory."""
        trajectories = self.trajectories(split)
        if not trajectories:
            return None
        return WindowDataset(
            trajectories,
            window=window,
            num_points=self.num_points,
            fields=self.fields,
            field_normalizer=self.field_normalizer,
            param_normalizer=self.param_normalizer,
            sampling=self.sampling,
            stride=self.window_stride,
        )

    def frame_dataset(
        self,
        trajectory: Trajectory,
        *,
        start: int = 0,
        end: int | None = None,
        node_indices: np.ndarray | None = None,
    ) -> FrameDataset:
        """Streaming per-frame dataset for one trajectory (evaluation)."""
        return FrameDataset(
            trajectory,
            fields=self.fields,
            field_normalizer=self.field_normalizer,
            start=start,
            end=end,
            node_indices=node_indices,
        )

    def increment_scale(self) -> torch.Tensor:
        """Per-field std of the frame-to-frame increment in *normalized* units, ``(C,)``.

        Used to express the residual target in units of its own spread so that every
        field weighs equally in the loss (``task.scale_residual``). Needs ``stats.yaml``.
        """
        from turtles.data.normalization import load_stats

        stats_path = Path(
            str(self.cfg.get("stats_path") or self.data_dir / STATS_FILENAME)
        ).expanduser()
        stats = load_stats(stats_path)
        delta = np.array([stats["std_delta"][f] for f in self.fields], dtype=np.float32)
        scale = self.field_normalizer.scale
        if scale is not None:
            delta = delta / scale
        return torch.from_numpy(delta)

    def params_tensor(self, trajectory: Trajectory) -> torch.Tensor | None:
        """``(1, P)`` normalized parameter tensor, or ``None`` when no parameter is used."""
        if not self.param_names:
            return None
        return torch.from_numpy(self.param_normalizer(trajectory.params)).unsqueeze(0)

    def check_batchable(self, split: str, batch_size: int) -> int | None:
        """Make samples collatable: clamp ``num_points`` to the smallest mesh of ``split``.

        A sample whose mesh is smaller than ``num_points`` would return all its nodes and
        could not be stacked with fixed-size samples. With ``batch_size > 1`` the cap is
        lowered to the smallest mesh (with a warning); the new value is returned.
        """
        if batch_size <= 1 or self.num_points is None:
            return self.num_points
        smallest = min((t.num_points for t in self.trajectories(split)), default=None)
        if smallest is not None and smallest < int(self.num_points):
            logger.warning(
                "data.num_points=%d exceeds the smallest mesh of split '%s' (%d nodes); "
                "clamping to %d so that batches can be collated",
                self.num_points,
                split,
                smallest,
                smallest,
            )
            self.num_points = int(smallest)
        return self.num_points

    def summary(self) -> str:
        """One-line description for logs."""
        parts = [
            f"root={self.data_dir}",
            f"subset={self.subset.name}",
            f"fields={self.fields}",
            f"params={self.param_names}",
            f"normalization={self.field_normalizer.strategy}",
            f"num_points={self.num_points}",
        ]
        return " ".join(parts)
