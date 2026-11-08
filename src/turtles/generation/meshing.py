"""Thin wrappers around Gmsh and the PyFR command line (import, partition)."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_mesh(geo_path: str | Path, msh_path: str | Path, *, verbosity: int = 0) -> Path:
    """Run Gmsh (Python API) on a ``.geo`` file and write the 3-D ``.msh``."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", verbosity)
        gmsh.open(str(geo_path))
        gmsh.model.mesh.generate(3)
        gmsh.write(str(msh_path))
    finally:
        gmsh.finalize()
    logger.info("Meshed %s -> %s", geo_path, msh_path)
    return Path(msh_path)


def import_mesh(msh_path: str | Path, pyfrm_path: str | Path) -> Path:
    """``pyfr import``: Gmsh mesh -> PyFR mesh."""
    subprocess.run(["pyfr", "import", str(msh_path), str(pyfrm_path)], check=True)
    logger.info("Imported %s -> %s", msh_path, pyfrm_path)
    return Path(pyfrm_path)


def partition_mesh(
    pyfrm_path: str | Path,
    out_dir: str | Path,
    *,
    parts: int = 4,
    scotch_library: str | None = None,
    element_weights: tuple[str, ...] = ("hex:3", "pri:2"),
) -> Path:
    """``pyfr partition`` with SCOTCH into ``parts`` sub-domains (one per GPU)."""
    env = dict(os.environ)
    if scotch_library:
        env["PYFR_SCOTCH_LIBRARY_PATH"] = scotch_library
        lib_dir = str(Path(scotch_library).parent)
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["pyfr", "partition", "-p", "scotch", str(parts), str(pyfrm_path), str(out_dir)]
    for weight in element_weights:
        cmd += ["-e", weight]
    subprocess.run(cmd, check=True, env=env)
    logger.info("Partitioned %s into %d parts under %s", pyfrm_path, parts, out_dir)
    return out_dir
