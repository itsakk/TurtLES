"""Download TurtLES (or a MINI-TurtLES subset) from the Hugging Face hub.

The hub repository mirrors the on-disk layout under a top-level ``data/`` folder::

    data/stats.yaml
    data/{train,val,test}/<case>/<case>_{coords,b0_99,...}.h5 (+ <case>_template.vtu)
    data/reference/Re1000_alpha0_a0p5_n2/...

Downloads are resumable and idempotent: files already present under ``root`` are
skipped, so ``ensure_available`` is cheap to call at the start of every run.
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Iterable
from pathlib import Path

from turtles.data.normalization import STATS_FILENAME
from turtles.data.subsets import REFERENCE_SPLIT, SPLITS, Subset

logger = logging.getLogger(__name__)

DEFAULT_REPO = "turtlesprojectml/turtles"


def resolve_data_dir(root: str | Path) -> Path:
    """Return the directory holding ``stats.yaml`` and the split folders.

    Accepts either that directory itself or a mirror of the hub repository (which
    nests everything under ``data/``).
    """
    root = Path(root).expanduser()
    nested = root / "data"
    markers = (STATS_FILENAME, *SPLITS, REFERENCE_SPLIT)
    if any((nested / marker).exists() for marker in markers):
        return nested
    return root


def download(
    root: str | Path,
    subset: Subset,
    splits: Iterable[str] = SPLITS,
    *,
    repo_id: str = DEFAULT_REPO,
    include_vtu: bool = False,
    revision: str | None = None,
    max_workers: int = 8,
    token: str | None = None,
) -> Path:
    """Download the files of ``subset`` for ``splits`` into ``root``.

    Returns the resolved data directory (``root/data``). Existing files are skipped.
    """
    from huggingface_hub import snapshot_download

    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    patterns = subset.hf_patterns(splits, include_vtu=include_vtu)
    logger.info("Downloading %s subset '%s' (%s) into %s", repo_id, subset.name, list(splits), root)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=patterns,
        revision=revision,
        max_workers=max_workers,
        token=token,
    )
    return resolve_data_dir(root)


def _split_complete(split_dir: Path, subset: Subset) -> bool:
    if not split_dir.is_dir():
        return False
    found = False
    for case_dir in split_dir.iterdir():
        if not case_dir.is_dir() or not subset.accepts(case_dir.name):
            continue
        found = True
        name = case_dir.name
        needed = [f"{name}_coords.h5", *(f"{name}_{s}.h5" for s in subset.chunk_suffixes)]
        if not all((case_dir / f).exists() for f in needed):
            return False
    return found


def is_available(data_dir: str | Path, subset: Subset, splits: Iterable[str] = SPLITS) -> bool:
    """Whether ``data_dir`` holds the statistics and complete cases of every split."""
    data_dir = Path(data_dir)
    if not (data_dir / STATS_FILENAME).exists():
        return False
    return all(_split_complete(data_dir / subset.split_dir_name(s), subset) for s in splits)


def ensure_available(
    root: str | Path,
    subset: Subset,
    splits: Iterable[str] = SPLITS,
    *,
    repo_id: str = DEFAULT_REPO,
    **download_kwargs,
) -> Path:
    """Return the data directory under ``root``, downloading missing files first."""
    splits = tuple(splits)
    data_dir = resolve_data_dir(root)
    if is_available(data_dir, subset, splits):
        return data_dir
    return download(root, subset, splits, repo_id=repo_id, **download_kwargs)


def remote_files(
    subset: Subset,
    splits: Iterable[str] = SPLITS,
    *,
    repo_id: str = DEFAULT_REPO,
    include_vtu: bool = False,
    revision: str | None = None,
    token: str | None = None,
) -> list[tuple[str, int]]:
    """List ``(path, size_in_bytes)`` of the hub files matched by the subset."""
    from huggingface_hub import HfApi

    patterns = subset.hf_patterns(splits, include_vtu=include_vtu)
    api = HfApi(token=token)
    entries = api.list_repo_tree(repo_id, repo_type="dataset", recursive=True, revision=revision)
    matched: list[tuple[str, int]] = []
    for entry in entries:
        path = getattr(entry, "path", None)
        size = getattr(entry, "size", None)
        if path is None or size is None:
            continue
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            matched.append((path, int(size)))
    return matched
