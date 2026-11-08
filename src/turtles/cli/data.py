"""``turtles-data``: download, inspect and benchmark the dataset.

turtles-data download --root data/turtles --subset G1-T100 --splits train val test
turtles-data size --subset G2-T200
turtles-data info --root data/turtles
turtles-data stats --root data/turtles --split train
turtles-data benchmark --root data/turtles --num-points 524288 --workers 8
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import time
from pathlib import Path

from turtles.data.hub import DEFAULT_REPO, download, remote_files, resolve_data_dir
from turtles.data.io import FIELDS, discover_trajectories
from turtles.data.normalization import STATS_FILENAME
from turtles.data.subsets import SPLITS, parse_subset

logger = logging.getLogger("turtles.data")


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _default_root() -> str:
    return os.environ.get("TURTLES_DATA_DIR", "data/turtles")


def cmd_download(args: argparse.Namespace) -> None:
    """Download a subset of the dataset from the hub."""
    if args.stats_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=args.root,
            allow_patterns=["data/stats.yaml"],
        )
        print(f"stats.yaml available under {resolve_data_dir(args.root)}")
        return
    subset = parse_subset(args.subset)
    data_dir = download(
        args.root,
        subset,
        args.splits,
        repo_id=args.repo,
        include_vtu=args.vtu,
        max_workers=args.workers,
    )
    print(f"Subset {subset.name} ({', '.join(args.splits)}) available under {data_dir}")


def cmd_size(args: argparse.Namespace) -> None:
    """Estimate the download size of a subset."""
    subset = parse_subset(args.subset)
    files = remote_files(subset, args.splits, repo_id=args.repo, include_vtu=args.vtu)
    total = sum(size for _, size in files)
    cases = {Path(path).parent for path, _ in files if path.endswith(".h5")}
    print(
        f"Subset {subset.name}, splits {', '.join(args.splits)}: {len(files)} files, "
        f"{len(cases)} trajectories, {_human(total)}"
    )


def cmd_info(args: argparse.Namespace) -> None:
    """Summarize what is present locally."""
    data_dir = resolve_data_dir(args.root)
    subset = parse_subset(args.subset)
    print(f"Data directory: {data_dir}")
    print(f"stats.yaml    : {'present' if (data_dir / STATS_FILENAME).exists() else 'MISSING'}")
    for split in (*SPLITS, "reference"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        trajectories = discover_trajectories(
            split_dir, subset=None if split == "reference" else subset
        )
        frames = sorted({t.num_frames for t in trajectories})
        sample = trajectories[: args.probe]
        nodes = [t.num_points for t in sample]
        node_info = f", nodes {min(nodes):,}-{max(nodes):,} (first {len(sample)})" if nodes else ""
        print(f"{split:9s}: {len(trajectories):4d} trajectories, frames {frames}{node_info}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Recompute stats.yaml from a split."""
    from turtles.data.stats import compute_stats, write_stats

    data_dir = resolve_data_dir(args.root)
    trajectories = discover_trajectories(data_dir / args.split, limit=args.limit)
    if not trajectories:
        raise SystemExit(f"No trajectory in {data_dir / args.split}")
    stats = compute_stats(trajectories, FIELDS, frame_stride=args.stride, max_nodes=args.max_nodes)
    out = Path(args.out) if args.out else data_dir / STATS_FILENAME
    write_stats(stats, out)
    print(f"Wrote {out}")
    for key, values in stats.items():
        print(f"  {key:11s} " + " ".join(f"{f}={v:.4g}" for f, v in values.items()))


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Measure DataLoader throughput."""
    import torch

    from turtles.data.dataset import WindowDataset, build_loader
    from turtles.data.normalization import FieldNormalizer

    data_dir = resolve_data_dir(args.root)
    trajectories = discover_trajectories(data_dir / args.split, limit=args.limit)
    dataset = WindowDataset(
        trajectories,
        window=2,
        num_points=args.num_points,
        fields=FIELDS,
        field_normalizer=FieldNormalizer.identity(),
        sampling=args.sampling,
    )
    loader = build_loader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        prefetch_factor=args.prefetch, pin_memory=torch.cuda.is_available(),
    )  # fmt: skip
    print(
        f"{len(dataset):,} windows over {len(trajectories)} trajectories; "
        f"batch={args.batch_size} workers={args.workers} num_points={args.num_points}"
    )
    times: list[float] = []
    last = time.perf_counter()
    for i, batch in enumerate(loader):
        now = time.perf_counter()
        times.append(now - last)
        last = now
        if i == 0:
            mb = batch["fields"].numel() * 4 / 1e6
            print(
                f"first batch {times[0] * 1e3:.0f} ms, fields {tuple(batch['fields'].shape)} ({mb:.0f} MB)"
            )
        if i >= args.batches:
            break
    steady = times[1:]
    if steady:
        mean = statistics.mean(steady)
        print(
            f"steady state: mean {mean * 1e3:.0f} ms/batch, median {statistics.median(steady) * 1e3:.0f} ms, "
            f"{args.batch_size * args.num_points / mean / 1e6:.1f} M nodes/s"
        )


def build_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    parser = argparse.ArgumentParser(
        prog="turtles-data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_root(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", default=_default_root(), help="local data root (TURTLES_DATA_DIR)")

    p = sub.add_parser("download", help="download a subset from the Hugging Face hub")
    add_root(p)
    p.add_argument("--subset", default="full")
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--vtu", action="store_true", help="also fetch the *_template.vtu files")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stats-only", action="store_true", help="only fetch data/stats.yaml")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("size", help="estimate the download size of a subset")
    p.add_argument("--subset", default="full")
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--vtu", action="store_true")
    p.set_defaults(func=cmd_size)

    p = sub.add_parser("info", help="summarize the local data directory")
    add_root(p)
    p.add_argument("--subset", default="full")
    p.add_argument("--probe", type=int, default=3, help="trajectories whose mesh size is read")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("stats", help="recompute stats.yaml")
    add_root(p)
    p.add_argument("--split", default="train")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--max-nodes", type=int, default=1_000_000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("benchmark", help="DataLoader throughput")
    add_root(p)
    p.add_argument("--split", default="train")
    p.add_argument("--num-points", type=int, default=524_288)
    p.add_argument("--sampling", default="contiguous", choices=["contiguous", "random"])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per hub request otherwise
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
