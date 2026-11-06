"""Case setup and simulation stages.

A :class:`PipelineConfig` (loaded from a YAML file, see ``configs/generation/pipeline.yaml``)
holds the directories, the MPI launch command and the physical constants of the
campaign. :func:`setup_case` renders the PyFR input files in ``templates/`` for one
:class:`CaseParams`, stores the settings in the case directory and optionally builds the
mesh. :func:`run_stage` then runs one stage (``transient``, ``extend``, ``trajectory``,
``convert``) of a case from its directory alone, the same way on a workstation or inside
a batch job.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from turtles.data.cases import CaseParams
from turtles.generation import geometry, meshing

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
CASE_FILE = "case.yaml"  # parameters and split of the case
PIPELINE_FILE = "pipeline.yaml"  # settings the case was set up with
STAGES = ("transient", "extend", "trajectory", "convert")

# Freestream velocity for Mach 0.2 in a gas with gamma = 1.4, rho = p = 1.
FREESTREAM_VELOCITY = 0.2 * math.sqrt(1.4)
SPANWISE_PROBES = 64


@dataclass
class PipelineConfig:
    """Directories, solver launch settings and campaign constants."""

    # Directories.
    cases_dir: str = "cases"
    trajectories_dir: str = "trajectories"
    dataset_dir: str = "dataset"
    plots_dir: str = "transient_plots"
    # Solver runs: one MPI rank per GPU; the mesh is partitioned into ``gpus`` parts.
    gpus: int = 4
    mpi_launcher: str = "mpirun -n {gpus}"  # e.g. "srun --mpi=pmi2" inside a SLURM job
    scotch_library: str | None = None
    # Campaign constants.
    flow_through_time: float = 136.0
    initial_flow_through_times: int = 4
    extend_flow_through_times: int = 5
    trajectory_dt: float = 0.2
    trajectory_snapshots: int = 400
    sampling_box: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-3.0, -3.5, 0.0),
        (10.0, 3.5, 2 * math.pi),
    )
    growth_rate: float = 1.021
    bl_thickness: float = 0.5
    solver_order: int = 4
    convert_subdivision: int = 4

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load from a YAML mapping (unknown keys raise)."""
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
        if "sampling_box" in data:
            lo, hi = data["sampling_box"]
            data["sampling_box"] = (tuple(lo), tuple(hi))
        return cls(**data)

    def to_yaml(self, path: str | Path) -> Path:
        """Write the config."""
        path = Path(path)
        data = asdict(self)
        data["sampling_box"] = [list(self.sampling_box[0]), list(self.sampling_box[1])]
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    @property
    def trajectory_duration(self) -> float:
        """Time units covered by the trajectory run."""
        return self.trajectory_dt * self.trajectory_snapshots


@dataclass(frozen=True)
class CaseLayout:
    """Paths of one case."""

    case_dir: Path
    trajectories_dir: Path
    dataset_dir: Path

    @property
    def solution_dir(self) -> Path:
        """Transient / extend solutions and probe data."""
        return self.case_dir / "solution"

    @property
    def partition_dir(self) -> Path:
        """Partitioned PyFR mesh."""
        return self.case_dir / "partitiondir"

    @property
    def geo(self) -> Path:
        """Gmsh geometry file."""
        return self.case_dir / "cylinder.geo"

    @property
    def msh(self) -> Path:
        """Gmsh mesh."""
        return self.case_dir / "cylinder.msh"

    @property
    def pyfrm(self) -> Path:
        """Imported PyFR mesh (unpartitioned)."""
        return self.case_dir / "cylinder.pyfrm"


def case_layout(cfg: PipelineConfig, case: CaseParams, split: str) -> CaseLayout:
    """Where the files of ``case`` live."""
    return CaseLayout(
        case_dir=Path(cfg.cases_dir) / case.name,
        trajectories_dir=Path(cfg.trajectories_dir) / case.name,
        dataset_dir=Path(cfg.dataset_dir) / split / case.name,
    )


def sampler_points(a: float, num_spanwise: int = SPANWISE_PROBES) -> str:
    """PyFR ``samp-pts`` entry: three centerline rakes at ``x = a + 4, 6.5, 9.5``."""
    z = [i * 2 * math.pi / num_spanwise for i in range(num_spanwise)]
    rows = []
    for offset in (4.0, 6.5, 9.5):
        x = a + offset
        rows.append(f"    # x = {x}")
        rows.append("    " + ", ".join(f"({x}, 0, {zz:.10f})" for zz in z) + ",")
    text = "\n".join(rows).rstrip(",")
    return "[\n" + text + "]"


def render_template(name: str, values: dict[str, Any]) -> str:
    """Fill ``templates/<name>`` with ``str.format`` placeholders."""
    return (TEMPLATE_DIR / name).read_text().format(**values)


def template_values(cfg: PipelineConfig, case: CaseParams, split: str) -> dict[str, Any]:
    """All placeholders used by the ``.ini`` templates."""
    layout = case_layout(cfg, case, split)
    u = FREESTREAM_VELOCITY
    ftt = cfg.flow_through_time
    transient_end = cfg.initial_flow_through_times * ftt
    lo, hi = cfg.sampling_box
    return {
        "simname": case.name,
        "split": split,
        "u": f"{u:.10f}",
        "mu": f"{u / case.Re:.9E}",
        "order": cfg.solver_order,
        "ftt": f"{ftt:g}",
        "transient_end": f"{transient_end:g}",
        "extend_duration": f"{cfg.extend_flow_through_times * ftt:g}",
        "trajectory_duration": f"{cfg.trajectory_duration:g}",
        "trajectory_dt": f"{cfg.trajectory_dt:g}",
        "box_lo": ", ".join(f"{v:g}" for v in lo),
        "box_hi": ", ".join(f"{v:g}" for v in hi),
        "sampler_points": sampler_points(case.a),
        "case_dir": str(layout.case_dir),
        "solution_dir": str(layout.solution_dir),
        "trajectories_dir": str(layout.trajectories_dir),
        "dataset_dir": str(layout.dataset_dir),
        "plots_dir": str(cfg.plots_dir),
    }


def setup_case(
    cfg: PipelineConfig,
    case: CaseParams,
    split: str,
    *,
    mesh: bool = True,
    overwrite: bool = False,
) -> CaseLayout | None:
    """Create the case directory with PyFR inputs and (optionally) the partitioned mesh.

    Returns ``None`` when the directory already exists and ``overwrite`` is False.
    """
    layout = case_layout(cfg, case, split)
    if layout.case_dir.exists() and not overwrite:
        logger.warning("%s exists; skipping (use overwrite=True to redo)", layout.case_dir)
        return None
    layout.solution_dir.mkdir(parents=True, exist_ok=True)
    layout.partition_dir.mkdir(parents=True, exist_ok=True)
    layout.trajectories_dir.mkdir(parents=True, exist_ok=True)

    values = template_values(cfg, case, split)
    for name in ("transient.ini", "extend.ini", "trajectory.ini"):
        (layout.case_dir / name).write_text(render_template(name, values))
    (layout.case_dir / CASE_FILE).write_text(
        yaml.safe_dump({**case.as_dict(), "split": split}, sort_keys=False)
    )
    cfg.to_yaml(layout.case_dir / PIPELINE_FILE)

    geometry.write_geo(
        layout.geo,
        Re=case.Re,
        inflow_angle=math.radians(case.alpha),
        a=case.a,
        b=0.5,
        n=case.n,
        growth_rate=cfg.growth_rate,
        bl_thickness=cfg.bl_thickness,
    )
    if mesh:
        meshing.generate_mesh(layout.geo, layout.msh)
        meshing.import_mesh(layout.msh, layout.pyfrm)
        meshing.partition_mesh(
            layout.pyfrm,
            layout.partition_dir,
            parts=cfg.gpus,
            scotch_library=cfg.scotch_library,
        )
    logger.info("Set up %s (%s)", layout.case_dir, split)
    return layout


def case_status(layout: CaseLayout) -> str:
    """Rough progress of a case from the files present on disk."""
    if any(layout.dataset_dir.glob("*_b300_399.h5")):
        return "converted"
    if any(layout.trajectories_dir.glob("*.pyfrs")):
        return "trajectory"
    if any(layout.solution_dir.glob("*.pyfrs")):
        return "transient"
    if (
        (layout.partition_dir / "cylinder.pyfrm").exists() or any(layout.partition_dir.iterdir())
        if layout.partition_dir.exists()
        else False
    ):
        return "meshed"
    return "setup" if layout.case_dir.exists() else "missing"


def load_case(case_dir: str | Path) -> tuple[PipelineConfig, CaseParams, CaseLayout]:
    """Settings, parameters and paths of a case directory created by :func:`setup_case`."""
    case_dir = Path(case_dir).resolve()
    cfg = PipelineConfig.from_yaml(case_dir / PIPELINE_FILE)
    meta = yaml.safe_load((case_dir / CASE_FILE).read_text())
    split = meta.pop("split")
    case = CaseParams(**meta)
    layout = case_layout(cfg, case, split)
    return cfg, case, CaseLayout(case_dir, layout.trajectories_dir, layout.dataset_dir)


def latest_solution(solution_dir: Path, simname: str) -> tuple[Path, str]:
    """Most recent ``<simname>_t<time>.pyfrs`` in ``solution_dir`` and its time string."""
    pattern = re.compile(rf"^{re.escape(simname)}_t([0-9.]+)\.pyfrs$")
    found = [
        (float(match.group(1)), match.group(1), path)
        for path in solution_dir.glob("*.pyfrs")
        if (match := pattern.match(path.name))
    ]
    if not found:
        raise FileNotFoundError(
            f"no {simname}_t*.pyfrs in {solution_dir}: run the transient stage first"
        )
    _, time, path = max(found)
    return path, time


def set_ini_times(ini: Path, t0: str, tend: str) -> None:
    """Set the ``t0`` and ``tend`` lines of a PyFR ``.ini`` file."""
    text = re.sub(r"(?m)^t0 =.*$", f"t0 = {t0}", ini.read_text())
    ini.write_text(re.sub(r"(?m)^tend =.*$", f"tend = {tend}", text))


def run_stage(
    case_dir: str | Path,
    stage: str,
    *,
    launcher: str | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Run one stage of a case; returns the command (the PyFR command for solver stages).

    ``transient`` starts PyFR from the initial condition; ``extend`` and ``trajectory``
    restart it from the latest solution in ``solution/`` (``extend`` continues the
    transient by ``extend_flow_through_times``, ``trajectory`` records the snapshots);
    ``convert`` writes the HDF5 files of the trajectory. ``launcher`` overrides
    ``mpi_launcher`` (``{gpus}`` is replaced by the number of GPUs). With ``dry_run`` the
    command is only returned and no file is changed.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    cfg, case, layout = load_case(case_dir)
    if stage == "convert":
        command = ["convert", str(layout.trajectories_dir), "->", str(layout.dataset_dir)]
        if not dry_run:
            from turtles.generation.convert import convert_case

            convert_case(
                trajectories_dir=layout.trajectories_dir,
                mesh_path=layout.partition_dir / "cylinder.pyfrm",
                prefix=case.name,
                output_dir=layout.dataset_dir,
                subdivision=cfg.convert_subdivision,
            )
        return command

    launch = shlex.split((launcher or cfg.mpi_launcher).format(gpus=cfg.gpus))
    pyfr = [sys.executable, "-u", "-m", "pyfr"]
    mesh = "partitiondir/cylinder.pyfrm"
    if stage == "transient":
        command = [*launch, *pyfr, "run", "-b", "cuda", mesh, "transient.ini"]
    else:
        duration = (
            cfg.extend_flow_through_times * cfg.flow_through_time
            if stage == "extend"
            else cfg.trajectory_duration
        )
        solution, t0 = latest_solution(layout.solution_dir, case.name)
        ini = f"{stage}.ini"
        if not dry_run:
            set_ini_times(layout.case_dir / ini, t0, f"{float(t0) + duration + 0.1:.3f}")
        start = str(solution.relative_to(layout.case_dir))
        command = [*launch, *pyfr, "-p", "restart", "-b", "cuda", mesh, start, ini]
    if not dry_run:
        logger.info("%s: %s", stage, shlex.join(command))
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        subprocess.run(command, cwd=layout.case_dir, env=env, check=True)
    return command
