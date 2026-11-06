"""Checkpoint files: ``last.pt`` for resuming, ``best.pt`` for evaluation.

A checkpoint is a dict with ``model`` (state dict of the raw module), ``optimizer``,
``scheduler``, ``step``, ``best_metric``, ``config`` (the resolved training config as a
plain dict) and ``normalizer`` (field normalization used at training time).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_PREFIXES = ("_orig_mod.", "module.")

LAST_NAME = "last.pt"
BEST_NAME = "best.pt"


def strip_prefixes(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove ``torch.compile`` / DDP wrapper prefixes from state-dict keys."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in _PREFIXES:
            while key.startswith(prefix):
                key = key[len(prefix) :]
        out[key] = value
    return out


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the raw module behind ``torch.compile`` and DDP wrappers."""
    module = model
    for _ in range(4):
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod
        elif hasattr(module, "module") and isinstance(module.module, torch.nn.Module):
            module = module.module
        else:
            break
    return module


def save_checkpoint(path: str | Path, state: Mapping[str, Any]) -> Path:
    """Atomically write ``state`` to ``path`` (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(dict(state), tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint` (trusted source: full unpickling)."""
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def load_model_weights(
    model: torch.nn.Module, state: Mapping[str, torch.Tensor], *, strict: bool = True
) -> None:
    """Load ``state`` into the raw module behind ``model``, tolerant to wrapper prefixes."""
    unwrap(model).load_state_dict(strip_prefixes(state), strict=strict)


def resolve_checkpoint(
    spec: str | Path, output_root: str | Path | None = None, *, prefer: str = BEST_NAME
) -> Path:
    """Turn a checkpoint spec into a file path.

    ``spec`` may be a ``.pt`` file, a run directory (``prefer`` then ``last.pt``) or a run
    name resolved under ``<output_root>/runs``.
    """
    candidate = Path(spec).expanduser()
    if candidate.is_file():
        return candidate
    run_dirs = [candidate]
    if output_root is not None:
        run_dirs.append(Path(output_root).expanduser() / "runs" / str(spec))
    for run_dir in run_dirs:
        if run_dir.is_dir():
            for name in (prefer, LAST_NAME, BEST_NAME):
                if (run_dir / name).is_file():
                    return run_dir / name
    raise FileNotFoundError(
        f"No checkpoint found for '{spec}' (looked in {[str(d) for d in run_dirs]})"
    )
