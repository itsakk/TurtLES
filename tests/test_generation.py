from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from turtles.data.cases import CaseParams
from turtles.generation.geometry import boundary_layer_cells, cell_sizes, generate_geo
from turtles.generation.pipeline import (
    PipelineConfig,
    case_status,
    load_case,
    run_stage,
    setup_case,
    template_values,
)
from turtles.generation.sampling import ParameterSpace, assign_split, read_design, write_design


def test_cell_sizes_interpolate_between_references():
    low, high = cell_sizes(1000), cell_sizes(3900)
    assert low["l_t"] == pytest.approx(0.051) and high["l_t"] == pytest.approx(0.022)
    mid = cell_sizes(2000)
    assert high["l_t"] < mid["l_t"] < low["l_t"]
    assert boundary_layer_cells(0.5, 0.05, 1.021) > 5


@pytest.mark.parametrize("n", [1.0, 2.0, 5.0, math.inf])
def test_generate_geo_contains_all_blocks(n):
    text = generate_geo(Re=2500, inflow_angle=math.radians(20), a=0.8, b=0.5, n=n)
    for token in (
        "Extrude",
        'Physical Surface("inlet")',
        "Transfinite Surface",
        "Background Field = 7",
        "n_z   = 30",
    ):
        assert token in text
    assert ("Line(1) = {" in text) == math.isinf(n)  # rectangles use straight edges


def test_parameter_space_transform_bounds():
    space = ParameterSpace()
    unit = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.999]])
    lo, hi = space.transform(unit)
    assert (lo.Re, lo.alpha, lo.n) == (
        1000,
        0,
        1.0,
    )  # rounding of 999.5 -> 1000 (banker's) keeps the bound
    assert lo.a == pytest.approx(1 / 6, abs=1e-3)
    assert (hi.Re, hi.alpha) == (3900, 45) and math.isinf(hi.n) and hi.a == pytest.approx(1.5)
    assert [assign_split(i) for i in range(10)] == ["train"] * 8 + ["val", "test"]


def test_sobol_design_roundtrip(tmp_path):
    pytest.importorskip("scipy")
    from turtles.generation.sampling import SobolDesign

    rows = SobolDesign(seed=42).generate(16)
    assert len(rows) == 16 and len({r.name for r in rows}) == 16
    assert all(1000 <= r.params.Re <= 3900 and 0 <= r.params.alpha <= 45 for r in rows)
    path = write_design(rows, tmp_path / "design.csv")
    back = read_design(path)
    assert [r.params for r in back] == [r.params for r in rows]
    assert [r.split for r in back] == [r.split for r in rows]


def _setup(tmp_path, **kwargs):
    cfg = PipelineConfig(
        cases_dir=str(tmp_path / "cases"),
        trajectories_dir=str(tmp_path / "traj"),
        dataset_dir=str(tmp_path / "dataset"),
        plots_dir=str(tmp_path / "plots"),
        **kwargs,
    )
    case = CaseParams(Re=1796, alpha=11, a=0.424, n=1)
    return cfg, case, setup_case(cfg, case, "train", mesh=False)


def test_setup_case_renders_templates(tmp_path):
    cfg, case, layout = _setup(tmp_path)
    assert layout is not None and layout.geo.exists()
    for name in ("transient.ini", "extend.ini", "trajectory.ini", "case.yaml", "pipeline.yaml"):
        assert "{simname}" not in (layout.case_dir / name).read_text(), name
    assert not list(layout.case_dir.glob("*.slurm"))
    transient = (layout.case_dir / "transient.ini").read_text()
    assert f"mu    = {0.2 * math.sqrt(1.4) / 1796:.9E}" in transient
    assert "tend = 544" in transient and "basename = Re1796_alpha11_a0p424_n1_t{t:.3f}" in transient
    assert f"({0.424 + 4}, 0, 0.0000000000)" in transient
    trajectory = (layout.case_dir / "trajectory.ini").read_text()
    assert (
        "region = box((-3, -3.5, 0), (10, 3.5, 6.28319))" in trajectory
        and "dt-out = 0.2" in trajectory
    )
    assert case_status(layout) == "setup"
    assert setup_case(cfg, case, "train", mesh=False) is None  # refuses to overwrite
    values = template_values(cfg, case, "train")
    assert values["trajectory_duration"] == "80"
    saved = tmp_path / "pipeline.yaml"
    cfg.to_yaml(saved)
    assert PipelineConfig.from_yaml(saved) == cfg
    loaded_cfg, loaded_case, loaded_layout = load_case(layout.case_dir)
    assert loaded_cfg == cfg and loaded_case == case
    assert loaded_layout.dataset_dir == layout.dataset_dir


def test_run_stage_commands_and_restart_times(tmp_path, monkeypatch):
    _, case, layout = _setup(tmp_path, mpi_launcher="srun --mpi=pmi2")
    pyfr = [sys.executable, "-u", "-m", "pyfr"]
    mesh = "partitiondir/cylinder.pyfrm"
    assert run_stage(layout.case_dir, "transient", dry_run=True) == [
        "srun", "--mpi=pmi2", *pyfr, "run", "-b", "cuda", mesh, "transient.ini"
    ]  # fmt: skip
    with pytest.raises(FileNotFoundError, match="transient stage first"):
        run_stage(layout.case_dir, "extend", dry_run=True)
    for t in ("136.000", "544.000", "1224.000"):  # natural order, not string order
        (layout.solution_dir / f"{case.name}_t{t}.pyfrs").touch()
    (layout.solution_dir / "other_t9999.000.pyfrs").touch()
    restart = ["-p", "restart", "-b", "cuda", mesh, f"solution/{case.name}_t1224.000.pyfrs"]
    command = run_stage(layout.case_dir, "extend", launcher="mpirun -n {gpus}", dry_run=True)
    assert command == ["mpirun", "-n", "4", *pyfr, *restart, "extend.ini"]
    assert "t0 = 0.0  #" in (layout.case_dir / "extend.ini").read_text()  # dry run: untouched

    # Real run with PyFR stubbed: the ini gets t0 + duration + 0.1 with three decimals.
    calls = []
    monkeypatch.setattr(
        "turtles.generation.pipeline.subprocess.run",
        lambda cmd, cwd, env, check: calls.append((cmd, cwd, env["OMP_NUM_THREADS"])),
    )
    run_stage(layout.case_dir, "extend")  # 5 flow-through times of 136
    run_stage(layout.case_dir, "trajectory")  # 400 snapshots every 0.2
    assert [c[0][-1] for c in calls] == ["extend.ini", "trajectory.ini"]
    assert calls[0][1] == layout.case_dir.resolve() and calls[0][2] == "1"
    extend = (layout.case_dir / "extend.ini").read_text()
    assert "\nt0 = 1224.000\n" in extend and "\ntend = 1904.100\n" in extend
    trajectory = (layout.case_dir / "trajectory.ini").read_text()
    assert "\nt0 = 1224.000\n" in trajectory and "\ntend = 1304.100\n" in trajectory
    assert trajectory.count("t0 =") == 1 and "set by" not in trajectory
    with pytest.raises(ValueError, match="unknown stage"):
        run_stage(layout.case_dir, "mesh", dry_run=True)


def test_convergence_statistic(tmp_path):
    pd = pytest.importorskip("pandas")
    from turtles.generation.convergence import compute_chi, convergence_history

    assert compute_chi([1, 1, 1, 1], [0.1, 0.1, 0.1, 0.1], mean_is_zero=False) == 0.0
    assert compute_chi([1, 1, 1.1, 1], [0.1, 0.1, 0.1, 0.1], mean_is_zero=False) == pytest.approx(
        10.0, rel=1e-6
    )
    assert compute_chi([5, 5, 5, 5], [0.1, 0.1, 0.1, 0.2], mean_is_zero=True) == pytest.approx(
        50.0, rel=1e-6
    )

    rng = np.random.default_rng(0)
    times = np.arange(0.0, 400.0, 2.0)
    rows = []
    for x in (4.5, 7.0, 10.0):
        for z in (0.0, 1.0, 2.0):
            rows.append(
                pd.DataFrame(
                    {
                        "t": times,
                        "x": x,
                        "y": 0.0,
                        "z": z,
                        "u": 0.2 + 0.01 * rng.standard_normal(len(times)),
                        "v": 0.01 * rng.standard_normal(len(times)),
                    }
                )
            )
    csv = tmp_path / "transient_data.csv"
    pd.concat(rows).to_csv(csv, index=False)
    history = convergence_history(csv, num_times=5)
    assert set(history.columns) == {"time", "x", "chi_u", "chi_v"}
    assert len(history) == 15 and (history["chi_u"] < 100).all()
