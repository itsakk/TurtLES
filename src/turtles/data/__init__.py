"""Data access: trajectories on disk or on the Hugging Face hub, datasets and normalizers."""

from __future__ import annotations

from turtles.data.cases import PARAM_NAMES, CaseParams, format_case_name, parse_case_name
from turtles.data.dataset import FrameDataset, TurtlesData, WindowDataset, build_loader
from turtles.data.hub import DEFAULT_REPO, download, ensure_available, resolve_data_dir
from turtles.data.io import FIELDS, Trajectory, discover_trajectories, read_coords, read_frame
from turtles.data.normalization import FieldNormalizer, ParamNormalizer
from turtles.data.subsets import FRAMES_PER_CHUNK, TOTAL_FRAMES, Subset, parse_subset

__all__ = [
    "DEFAULT_REPO",
    "FIELDS",
    "FRAMES_PER_CHUNK",
    "PARAM_NAMES",
    "TOTAL_FRAMES",
    "CaseParams",
    "FieldNormalizer",
    "FrameDataset",
    "ParamNormalizer",
    "Subset",
    "Trajectory",
    "TurtlesData",
    "WindowDataset",
    "build_loader",
    "discover_trajectories",
    "download",
    "ensure_available",
    "format_case_name",
    "parse_case_name",
    "parse_subset",
    "read_coords",
    "read_frame",
    "resolve_data_dir",
]
