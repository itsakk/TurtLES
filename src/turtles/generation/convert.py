"""Convert PyFR trajectory snapshots (``.pyfrs``) to the TurtLES HDF5 layout.

Every snapshot is exported to a high-order ``.vtu`` with ``pyfr export``, read with
PyVista, deduplicated (``clean``: solution points shared by neighbouring elements)
and appended to 100-frame chunk files ``<case>_b{start}_{end}.h5`` as a group
``sol_t{index}`` with datasets ``rho, p, u, v, w``. Node coordinates go to
``<case>_coords.h5`` once (and are re-checked periodically); a template ``.vtu``
with the mesh topology is kept for visualisation.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np

from turtles.data.subsets import FRAMES_PER_CHUNK, chunk_suffix

logger = logging.getLogger(__name__)

_TIME_RE = re.compile(r"ts([\d.]+)\.pyfrs$")


def _snapshot_time(path: Path) -> float:
    match = _TIME_RE.search(path.name)
    return float(match.group(1)) if match else 0.0


def list_snapshots(trajectories_dir: str | Path, prefix: str) -> list[Path]:
    """``.pyfrs`` snapshots of ``prefix`` sorted by simulation time."""
    files = [p for p in Path(trajectories_dir).glob("*.pyfrs") if f"{prefix}_ts" in p.name]
    return sorted(files, key=_snapshot_time)


def export_vtu(
    mesh_path: str | Path, snapshot: str | Path, vtu_path: str | Path, *, subdivision: int = 4
) -> Path:
    """``pyfr export`` one snapshot to a single-precision ``.vtu``."""
    subprocess.run(
        [
            "pyfr",
            "export",
            "-p",
            "single",
            "-k",
            str(subdivision),
            str(mesh_path),
            str(snapshot),
            str(vtu_path),
        ],
        check=True,
        capture_output=True,
    )
    return Path(vtu_path)


def convert_case(
    *,
    trajectories_dir: str | Path,
    mesh_path: str | Path,
    prefix: str,
    output_dir: str | Path,
    subdivision: int = 4,
    clean_tolerance: float = 1e-5,
    verify_every: int = 50,
    keep_vtu: bool = False,
) -> Path:
    """Convert all snapshots of one case; returns the output directory."""
    import pyvista as pv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = list_snapshots(trajectories_dir, prefix)
    if not snapshots:
        raise FileNotFoundError(f"no {prefix}_ts*.pyfrs under {trajectories_dir}")
    coords_path = output_dir / f"{prefix}_coords.h5"

    for index, snapshot in enumerate(snapshots):
        vtu_path = output_dir / f"{prefix}_ts{index}.vtu"
        if not vtu_path.exists():
            export_vtu(mesh_path, snapshot, vtu_path, subdivision=subdivision)
        grid = pv.read(vtu_path).clean(tolerance=clean_tolerance)
        points = np.asarray(grid.points)

        if index == 0:
            template = grid.copy()
            template.point_data["Density"] = np.ones(template.n_points, dtype=np.float32)
            template.point_data["Pressure"] = np.ones(template.n_points, dtype=np.float32)
            template.point_data["Velocity"] = np.ones((template.n_points, 3), dtype=np.float32)
            template.save(output_dir / f"{prefix}_template.vtu")
            with h5py.File(coords_path, "w") as handle:
                for axis, column in zip(("x", "y", "z"), range(3), strict=True):
                    handle.create_dataset(axis, data=points[:, column], compression="gzip")
            logger.info("Wrote %s (%d nodes)", coords_path, points.shape[0])
        elif index % verify_every == 0 or index == 20:
            with h5py.File(coords_path, "r") as handle:
                reference = np.column_stack([handle["x"][()], handle["y"][()], handle["z"][()]])
            if reference.shape != points.shape or not np.allclose(points, reference, atol=1e-8):
                raise ValueError(f"mesh changed at snapshot {index} of {prefix}")

        chunk_path = output_dir / f"{prefix}_{chunk_suffix(index // FRAMES_PER_CHUNK)}.h5"
        with h5py.File(chunk_path, "a") as handle:
            group_name = f"sol_t{index}"
            if group_name in handle:
                del handle[group_name]
            group = handle.create_group(group_name)
            group.create_dataset("rho", data=grid.point_data["Density"], compression="gzip")
            group.create_dataset("p", data=grid.point_data["Pressure"], compression="gzip")
            velocity = np.asarray(grid.point_data["Velocity"])
            for column, name in enumerate(("u", "v", "w")):
                group.create_dataset(name, data=velocity[:, column], compression="gzip")
        if not keep_vtu:
            vtu_path.unlink()
        logger.info("[%d/%d] %s -> %s", index + 1, len(snapshots), snapshot.name, chunk_path.name)
    return output_dir
