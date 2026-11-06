"""HDF5 access to TurtLES trajectories.

A trajectory directory holds one coordinates file (``<case>_coords.h5`` with
``x``, ``y``, ``z`` datasets) and up to four 100-frame solution chunks
(``<case>_b0_99.h5`` ... ``<case>_b300_399.h5``). Frame ``t`` lives in group
``sol_t{t}`` of chunk ``t // 100`` as one 1-D dataset per field.

All readers return ``float32`` NumPy arrays and accept an optional node selection
(a ``slice`` for a sequential read, or a sorted index array).
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path

import h5py
import numpy as np

from turtles.data.cases import CaseParams, parse_case_name
from turtles.data.subsets import FRAMES_PER_CHUNK, Subset

logger = logging.getLogger(__name__)

FIELDS: tuple[str, ...] = ("p", "rho", "u", "v", "w")
COORD_AXES: tuple[str, ...] = ("x", "y", "z")

# A slice, a few slices read one after the other (strided sampling), sorted indices or all.
Selection = slice | tuple[slice, ...] | np.ndarray | None

# Read a multi-slice selection contiguously when it keeps at least 1 node in this many.
_DENSE_READ_FACTOR = 4

_CHUNK_RE = re.compile(r"_b(\d+)_(\d+)\.h5$")


def frame_group(t: int) -> str:
    """Name of the HDF5 group holding frame ``t``."""
    return f"sol_t{t}"


def _scan_directory(directory: Path) -> tuple[Path, tuple[Path, ...]]:
    coords: Path | None = None
    chunks: dict[int, Path] = {}
    for path in directory.iterdir():
        if path.suffix != ".h5" or path.name.startswith("._"):
            continue
        if path.name.endswith("_coords.h5"):
            coords = path
            continue
        match = _CHUNK_RE.search(path.name)
        if match and int(match.group(1)) % FRAMES_PER_CHUNK == 0:
            chunks[int(match.group(1)) // FRAMES_PER_CHUNK] = path
    if coords is None:
        raise FileNotFoundError(f"No *_coords.h5 file in {directory}")
    if not chunks:
        raise FileNotFoundError(f"No solution chunk (*_b{{start}}_{{end}}.h5) in {directory}")
    ordered = [chunks[i] for i in range(len(chunks)) if i in chunks]
    if len(ordered) != len(chunks):
        raise ValueError(
            f"Solution chunks in {directory} are not contiguous from frame 0: "
            f"{sorted(p.name for p in chunks.values())}"
        )
    return coords, tuple(ordered)


class Trajectory:
    """One simulation case on disk: coordinates plus consecutive 100-frame chunks.

    Args:
        directory: Case directory named ``Re{Re}_alpha{alpha}_a{a}_n{n}``.
        max_frames: Cap on the number of usable frames (temporal subsets).
    """

    def __init__(self, directory: str | Path, *, max_frames: int | None = None):
        self.directory = Path(directory)
        self.name = self.directory.name
        self.params: CaseParams = parse_case_name(self.name)
        self.coords_path, self.chunk_paths = _scan_directory(self.directory)
        available = FRAMES_PER_CHUNK * len(self.chunk_paths)
        self.num_frames = min(available, max_frames) if max_frames else available
        self._num_points: int | None = None

    @property
    def num_points(self) -> int:
        """Number of mesh nodes (read once from the coordinates file)."""
        if self._num_points is None:
            with h5py.File(self.coords_path, "r") as handle:
                self._num_points = int(handle["x"].shape[0])
        return self._num_points

    def chunk_path(self, t: int) -> Path:
        """Path of the chunk file holding frame ``t``."""
        if not 0 <= t < self.num_frames:
            raise IndexError(f"frame {t} out of range for {self.name} ({self.num_frames} frames)")
        return self.chunk_paths[t // FRAMES_PER_CHUNK]

    def __repr__(self) -> str:
        return f"Trajectory({self.name}, frames={self.num_frames})"


class _HandleCache:
    """Per-process LRU cache of open ``h5py.File`` handles.

    Handles are not fork-safe, so a child process (DataLoader worker) drops the
    handles it inherited and reopens files on demand. The bound keeps the per-dataset
    chunk caches of h5py from accumulating gigabytes of RAM over long runs.
    """

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()
        self._pid = os.getpid()

    def get(self, path: str | Path) -> h5py.File:
        if os.getpid() != self._pid:
            self._handles = OrderedDict()
            self._pid = os.getpid()
        key = str(path)
        handle = self._handles.pop(key, None)
        if handle is None:
            handle = h5py.File(key, "r")
        self._handles[key] = handle
        while len(self._handles) > self.capacity:
            _, oldest = self._handles.popitem(last=False)
            with contextlib.suppress(Exception):  # a stale handle must never fail the reader
                oldest.close()
        return handle

    def clear(self) -> None:
        for handle in self._handles.values():
            with contextlib.suppress(Exception):
                handle.close()
        self._handles.clear()


_HANDLES = _HandleCache()


def open_h5(path: str | Path) -> h5py.File:
    """Return a cached read-only handle for ``path`` (see :class:`_HandleCache`)."""
    return _HANDLES.get(path)


def close_all_h5() -> None:
    """Close every cached handle in the current process."""
    _HANDLES.clear()


def _read_1d(dataset: h5py.Dataset, sel: Selection) -> np.ndarray:
    if sel is None:
        return dataset[()]
    if isinstance(sel, slice):
        return dataset[sel]
    if isinstance(sel, tuple):
        first, last = sel[0].start, sel[-1].stop
        count = sum(len(range(part.start, part.stop, part.step or 1)) for part in sel)
        if last - first <= _DENSE_READ_FACTOR * count:
            # Dense selection: one contiguous read beats several strided hyperslabs.
            block = dataset[first:last]
            return np.concatenate(
                [block[part.start - first : part.stop - first : part.step] for part in sel]
            )
        return np.concatenate([dataset[part] for part in sel])
    # h5py fancy indexing is slow and demands increasing indices: read, then gather.
    return dataset[()][sel]


def read_coords(traj: Trajectory, sel: Selection = None) -> np.ndarray:
    """Node coordinates ``(N, 3)`` of a trajectory, optionally restricted to ``sel``."""
    handle = open_h5(traj.coords_path)
    columns = [_read_1d(handle[axis], sel) for axis in COORD_AXES]
    return np.stack(columns, axis=-1).astype(np.float32, copy=False)


def read_frame(
    traj: Trajectory,
    t: int,
    fields: Sequence[str] = FIELDS,
    sel: Selection = None,
) -> np.ndarray:
    """Fields of frame ``t`` as ``(N, C)`` in the order of ``fields``."""
    group = open_h5(traj.chunk_path(t))[frame_group(t)]
    out: np.ndarray | None = None
    for c, field in enumerate(fields):
        column = _read_1d(group[field], sel)
        if out is None:
            out = np.empty((column.shape[0], len(fields)), dtype=np.float32)
        out[:, c] = column
    assert out is not None, "fields must not be empty"
    return out


def read_frames(
    traj: Trajectory,
    times: Iterable[int],
    fields: Sequence[str] = FIELDS,
    sel: Selection = None,
) -> np.ndarray:
    """Stack of frames ``(T, N, C)`` for the given ``times`` and node selection."""
    return np.stack([read_frame(traj, t, fields, sel) for t in times], axis=0)


def discover_trajectories(
    split_dir: str | Path,
    *,
    subset: Subset | None = None,
    max_frames: int | None = None,
    limit: int | None = None,
) -> list[Trajectory]:
    """List the usable trajectories of a split directory, sorted by case name.

    Directories that are not complete cases (missing coordinates, missing or
    non-contiguous chunks, unparsable names) are skipped with a warning so that a
    partially downloaded split remains usable.
    """
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        return []
    trajectories: list[Trajectory] = []
    skipped: list[str] = []
    for case_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        if case_dir.name.startswith("."):
            continue
        if subset is not None and not subset.accepts(case_dir.name):
            continue
        try:
            trajectories.append(Trajectory(case_dir, max_frames=max_frames))
        except (FileNotFoundError, ValueError) as exc:
            skipped.append(f"{case_dir.name}: {exc}")
    if skipped:
        logger.warning(
            "Skipped %d incomplete case(s) in %s (first: %s)", len(skipped), split_dir, skipped[0]
        )
    if limit is not None:
        trajectories = trajectories[: int(limit)]
    return trajectories
