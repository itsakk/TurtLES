r"""``turtles-report``: aggregate rollout evaluations into the paper's tables and figure.

Example::

    turtles-report --eval UPT=outputs/runs/123-upt-full/eval/test \
                   --eval CALM-PDE=outputs/runs/124-calm_pde-full/eval/test \
                   --eval Persistence=outputs/runs/persistence-baseline/eval/test \
                   --out paper/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

from turtles.evaluation.report import build_report


def build_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    parser = argparse.ArgumentParser(
        prog="turtles-report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="display name and evaluation directory (repeatable, order kept)",
    )
    parser.add_argument("--out", type=Path, default=Path("report"))
    parser.add_argument("--formats", nargs="+", default=["md", "tex"], choices=["md", "tex"])
    parser.add_argument("--no-figure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = build_parser().parse_args(argv)
    evals: dict[str, Path] = {}
    for spec in args.eval:
        if "=" not in spec:
            raise SystemExit(f"--eval expects NAME=DIR, got '{spec}'")
        name, directory = spec.split("=", 1)
        evals[name.strip()] = Path(directory).expanduser()
    written = build_report(evals, args.out, formats=tuple(args.formats), figure=not args.no_figure)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
