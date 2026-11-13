#!/usr/bin/env python3
"""Download TurtLES: the full dataset, a mini-TurtLES subset, the debug subset, or single flow cases.

Requirements:
    Python >= 3.9 and huggingface_hub:   pip install -U huggingface_hub

Usage:
    python download_turtles.py --list                        # all subsets with their exact sizes
    python download_turtles.py TurtLES-debug                 # ~5 GB, for installation checks
    python download_turtles.py TurtLES-G1T100                # a mini-TurtLES subset
    python download_turtles.py TurtLES-full                  # everything (~14 TB)
    python download_turtles.py Re1089_alpha12_a1p302_n2      # one flow case (all 400 frames)

Options:
    --dir PATH          where to put the data (default: current directory)
    --split SPLITS      only these splits, e.g. --split test or --split train,val (subsets only)
    --with-vtu          also download each case's *_template.vtu (mesh template for ParaView)
    --with-spectra      also download each case's *_spectra.npz (velocity spectra at three wake probes)
    --yes               do not ask for confirmation (for batch jobs)

Subsets (see the dataset card):
    TurtLES-G{1-4}T{100,200,300,400}   geometry tier x temporal tier
        G4: all shape families (n = 1, 2, 5, inf)   G3: n = 1, 2, inf   G2: n = 2, inf   G1: n = 2
        T100 ... T400: the first 100 ... 400 frames of every trajectory
    TurtLES-full                       = TurtLES-G4T400 (all 400 cases, all frames)
    TurtLES-debug                      first 100 frames of the reference case Re1000_alpha0_a0p5_n2

The data keeps the repository layout (data/<split>/<case_id>/...), and cases.csv (the official
split file) is always downloaded too. Files that are already present are skipped, so moving up
a tier only fetches the difference, and an interrupted download resumes where it stopped.

Tips for large downloads:
    - Log in first (hf auth login): logged-in downloads have higher rate limits.
    - On clusters, set HF_HOME to scratch space; huggingface_hub keeps small metadata there.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import sys

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import RepoFile

REPO_ID = "turtlesprojectml/turtles"
SPLITS = ["train", "val", "test"]  # the reference case lives in data/reference/ and is not part of any subset
FAMILIES = {1: {"2"}, 2: {"2", "inf"}, 3: {"1", "2", "inf"}, 4: {"1", "2", "5", "inf"}}
CHUNKS = ["_b0_99.h5", "_b100_199.h5", "_b200_299.h5", "_b300_399.h5"]
COORDS, TEMPLATE, SPECTRA = "_coords.h5", "_template.vtu", "_spectra.npz"
REFERENCE_CASE = "Re1000_alpha0_a0p5_n2"
CASE_RE = re.compile(r"Re\d+_alpha\d+_a\d+p\d+_n(1|2|5|inf)")


def list_repo(repo_id: str) -> list[tuple[str, str, str, str, int]]:
    """All data files as (path, split, case_id, file suffix, size)."""
    out = []
    for f in HfApi().list_repo_tree(repo_id, repo_type="dataset", path_in_repo="data", recursive=True):
        if not isinstance(f, RepoFile):
            continue
        parts = f.path.split("/")  # data/<split>/<case_id>/<file>
        if len(parts) != 4 or not parts[3].startswith(parts[2]):
            continue
        out.append((f.path, parts[1], parts[2], parts[3][len(parts[2]):], f.size))
    return out


def subset_suffixes(t: int, extras: set[str]) -> set[str]:
    return {COORDS, *CHUNKS[: t // 100]} | extras


def select(name: str, files, splits: list[str], extras: set[str]):
    """Return the list of (path, size) for a subset or case name, or exit with a message."""
    low = name.lower()
    if low == "turtles-debug":
        want = {COORDS, CHUNKS[0]} | extras
        return [(p, s) for p, sp, c, suf, s in files if sp == "reference" and c == REFERENCE_CASE and suf in want]
    if CASE_RE.fullmatch(name):
        want = {COORDS, *CHUNKS} | extras
        sel = [(p, s) for p, sp, c, suf, s in files if c == name and suf in want]
        if not sel:
            sys.exit(f"Case {name!r} not found. See cases.csv for all case names.")
        return sel
    if low == "turtles-full":
        g, t = 4, 400
    else:
        m = re.fullmatch(r"turtles-g([1-4])t(100|200|300|400)", low)
        if not m:
            sys.exit(f"Unknown name {name!r}. Expected TurtLES-G{{1-4}}T{{100,200,300,400}}, TurtLES-full, "
                     f"TurtLES-debug, or a case name like Re1089_alpha12_a1p302_n2. Run with --list to see all.")
        g, t = int(m.group(1)), int(m.group(2))
    fams, want = FAMILIES[g], subset_suffixes(t, extras)
    return [(p, s) for p, sp, c, suf, s in files
            if sp in splits and c.rsplit("_n", 1)[1] in fams and suf in want]


def human(nbytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1000 or unit == "TB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes:.0f} B"
        nbytes /= 1000


def print_list(files, extras: set[str]) -> None:
    with_ = [label for suf, label in ((TEMPLATE, ".vtu files"), (SPECTRA, "spectra")) if suf in extras]
    print("Subset sizes (all splits" + (", with " + " and ".join(with_) if with_ else "") + "):\n")
    print(f"  {'':<16}" + "".join(f"{'T' + str(t):>11}" for t in (100, 200, 300, 400)))
    for g in (4, 3, 2, 1):
        row = [sum(s for _, s in select(f"TurtLES-G{g}T{t}", files, SPLITS, extras)) for t in (100, 200, 300, 400)]
        print(f"  {'TurtLES-G' + str(g) + 'T…':<16}" + "".join(f"{human(b):>11}" for b in row))
    debug = sum(s for _, s in select("TurtLES-debug", files, SPLITS, extras))
    print(f"\n  TurtLES-full  = TurtLES-G4T400\n  TurtLES-debug   {human(debug)}")
    print("\n  Single cases: pass the case name (see cases.csv).")


def _stop_now(signum, frame) -> None:
    sys.stderr.flush()
    print("\n\nInterrupted. Finished files are kept; run the same command again to continue.", flush=True)
    os._exit(130)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download TurtLES or a subset of it.",
                                 epilog="Run with --list to see all subsets and their sizes.")
    ap.add_argument("name", nargs="?", help="TurtLES-G{1-4}T{100..400}, TurtLES-full, TurtLES-debug, or a case name")
    ap.add_argument("--dir", default=".", help="target directory (default: current directory)")
    ap.add_argument("--split", default=",".join(SPLITS), help="comma-separated splits (default: train,val,test)")
    ap.add_argument("--with-vtu", action="store_true", help="also download the *_template.vtu files")
    ap.add_argument("--with-spectra", action="store_true", help="also download the *_spectra.npz files")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--list", action="store_true", help="list all subsets with their sizes and exit")
    args = ap.parse_args()

    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    if bad := [s for s in splits if s not in SPLITS]:
        sys.exit(f"Unknown split(s) {bad}. Choose from {SPLITS}.")
    if not args.list and not args.name:
        ap.print_help()
        sys.exit(1)

    extras = ({TEMPLATE} if args.with_vtu else set()) | ({SPECTRA} if args.with_spectra else set())

    print(f"Reading the file list of {REPO_ID} ...")
    files = list_repo(REPO_ID)
    if args.list:
        print_list(files, extras)
        return

    m = re.fullmatch(r"turtles-(g[1-4]t\d+|full|debug)", args.name.lower())
    if m:
        args.name = "TurtLES-" + m.group(1).upper().replace("FULL", "full").replace("DEBUG", "debug")
    is_subset = not CASE_RE.fullmatch(args.name) and args.name != "TurtLES-debug"
    if not is_subset and args.split != ",".join(SPLITS):
        print("Note: --split only applies to TurtLES-G…T… and TurtLES-full; ignored here.")
    selection = select(args.name, files, splits if is_subset else SPLITS, extras)
    if not selection:
        sys.exit("Nothing matches this selection.")

    total = sum(s for _, s in selection)
    todo = [(p, s) for p, s in selection
            if not (os.path.isfile(os.path.join(args.dir, p)) and os.path.getsize(os.path.join(args.dir, p)) == s)]
    remaining = sum(s for _, s in todo)
    n_cases = len({p.split("/")[2] for p, _ in selection})

    os.makedirs(args.dir, exist_ok=True)
    free = shutil.disk_usage(args.dir).free
    print(f"\n  Selection:  {args.name}" + (f"  (splits: {', '.join(splits)})" if is_subset else ""))
    print(f"  Cases:      {n_cases}   Files: {len(selection)}")
    print(f"  Size:       {human(total)}   still to download: {human(remaining)}")
    print(f"  Target:     {os.path.abspath(args.dir)}   free space: {human(free)}\n")

    if remaining > free:
        sys.exit("Not enough free disk space for this selection. Choose a smaller subset or another --dir.")
    if todo and not args.yes:
        if not sys.stdin.isatty():
            sys.exit("Not running interactively: add --yes to start the download.")
        if input("Proceed? [y/N] ").strip().lower() != "y":
            sys.exit("Aborted.")
    if not todo:
        print("All data files are already present.")
    signal.signal(signal.SIGINT, _stop_now)
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=args.dir,
                      allow_patterns=[p for p, _ in todo] + ["cases.csv"])
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sys.stderr.flush()  
    print(f"\n\nDone: {args.name}." + (" Please report this subset name alongside published results." if is_subset else ""),
          flush=True)


if __name__ == "__main__":
    main()
