"""``turtles-gen``: drive the data generation pipeline.

    turtles-gen sample  --n 1024 --out design.csv
    turtles-gen setup   --design design.csv --config pipeline.yaml --n-cases 50 [--no-mesh]
    turtles-gen status  --design design.csv --config pipeline.yaml
    turtles-gen run     transient|extend|trajectory|convert <case_dir> [--launcher CMD]
    turtles-gen check   <case_dir> [--plots-dir DIR]          # stationarity statistic chi
    turtles-gen convert <case_dir> --trajectories-dir D --output-dir D [--subdivision 4]

``setup`` writes the mesh, the PyFR input files and the settings into each case
directory; ``run`` executes one stage of a case from that directory, on this machine or
inside a batch job (see ``examples/slurm/generate_stage.sbatch``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from turtles.data.cases import parse_case_name
from turtles.generation.pipeline import (
    STAGES,
    PipelineConfig,
    case_layout,
    case_status,
    load_case,
    run_stage,
    setup_case,
)
from turtles.generation.sampling import SobolDesign, read_design, write_design

logger = logging.getLogger("turtles.generation")

DEFAULT_PIPELINE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "generation" / "pipeline.yaml"
)


def _load_config(path: str | None) -> PipelineConfig:
    return PipelineConfig.from_yaml(path or DEFAULT_PIPELINE_CONFIG)


def cmd_sample(args: argparse.Namespace) -> None:
    """Draw the Sobol design."""
    rows = SobolDesign(seed=args.seed).generate(args.n)
    path = write_design(rows, args.out)
    counts = {s: sum(r.split == s for r in rows) for s in ("train", "val", "test")}
    print(f"Wrote {len(rows)} cases to {path} ({counts})")


def cmd_setup(args: argparse.Namespace) -> None:
    """Create case directories (job files, geometry, optional mesh)."""
    cfg = _load_config(args.config)
    rows = read_design(args.design)
    created = 0
    for row in rows:
        if args.n_cases is not None and created >= args.n_cases:
            break
        layout = setup_case(
            cfg, row.params, row.split, mesh=not args.no_mesh, overwrite=args.overwrite
        )
        if layout is not None:
            created += 1
    print(f"Set up {created} case(s) under {cfg.cases_dir}")


def cmd_status(args: argparse.Namespace) -> None:
    """Print the stage reached by every case of the design."""
    cfg = _load_config(args.config)
    for row in read_design(args.design):
        layout = case_layout(cfg, row.params, row.split)
        print(f"{row.index:5d} {row.split:5s} {row.name:36s} {case_status(layout)}")


def report_stationarity(case_dir: Path, plots_dir: Path) -> None:
    """Print the latest stationarity statistic of every rake and save its history plot."""
    from turtles.generation.convergence import convergence_history, plot_convergence

    history = convergence_history(case_dir / "solution" / "transient_data.csv")
    latest = history[history["time"] == history["time"].max()]
    for _, row in latest.iterrows():
        print(f"x = {row['x']:.2f}: chi(u) = {row['chi_u']:.1f} %  chi(v) = {row['chi_v']:.1f} %")
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plot_convergence(history, plots_dir / f"{case_dir.name}_convergence.png")
    print(f"Wrote {path}")


def cmd_check(args: argparse.Namespace) -> None:
    """Compute and plot the stationarity statistic of a case."""
    case_dir = Path(args.case_dir)
    report_stationarity(case_dir, Path(args.plots_dir) if args.plots_dir else case_dir / "solution")


def cmd_run(args: argparse.Namespace) -> None:
    """Run one stage of a case (then report stationarity after transient / extend)."""
    command = run_stage(args.case_dir, args.stage, launcher=args.launcher, dry_run=args.dry_run)
    if args.dry_run:
        print(" ".join(command))
        return
    if args.stage in ("transient", "extend"):
        cfg, _, layout = load_case(args.case_dir)
        report_stationarity(layout.case_dir, Path(cfg.plots_dir))


def cmd_convert(args: argparse.Namespace) -> None:
    """Convert the trajectory snapshots of a case to HDF5."""
    from turtles.generation.convert import convert_case

    case_dir = Path(args.case_dir)
    prefix = parse_case_name(case_dir.name).name
    out = convert_case(
        trajectories_dir=args.trajectories_dir,
        mesh_path=case_dir / "partitiondir" / "cylinder.pyfrm",
        prefix=prefix,
        output_dir=args.output_dir,
        subdivision=args.subdivision,
        keep_vtu=args.keep_vtu,
    )
    print(f"Converted {prefix} -> {out}")


def build_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    parser = argparse.ArgumentParser(
        prog="turtles-gen",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample", help="draw the Sobol design")
    p.add_argument("--n", type=int, default=1024, help="number of cases (power of two)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="design.csv")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("setup", help="create case directories from the design")
    p.add_argument("--design", required=True)
    p.add_argument(
        "--config", default=None, help=f"pipeline YAML (default: {DEFAULT_PIPELINE_CONFIG.name})"
    )
    p.add_argument(
        "--n-cases", type=int, default=None, help="stop after creating this many new cases"
    )
    p.add_argument("--no-mesh", action="store_true", help="skip Gmsh / PyFR import / partition")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("status", help="stage reached by each case")
    p.add_argument("--design", required=True)
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("run", help="run one stage of a case")
    p.add_argument("stage", choices=STAGES)
    p.add_argument("case_dir")
    p.add_argument(
        "--launcher", default=None, help="MPI launch command (default: mpi_launcher of the case)"
    )
    p.add_argument("--dry-run", action="store_true", help="print the command, change nothing")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("check", help="stationarity statistic of the transient run")
    p.add_argument("case_dir")
    p.add_argument("--plots-dir", default=None)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("convert", help="convert .pyfrs snapshots to HDF5")
    p.add_argument("case_dir")
    p.add_argument("--trajectories-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subdivision", type=int, default=4)
    p.add_argument("--keep-vtu", action="store_true")
    p.set_defaults(func=cmd_convert)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
