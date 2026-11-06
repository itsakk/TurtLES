"""Shared helpers for the Hydra entry points."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("turtles")


def guarded(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run ``fn`` and print the traceback before re-raising.

    Launchers that forward the exception to another process (for example Hydra's
    submitit plugin) drop the traceback; printing it here keeps it in the job log.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        traceback.print_exc()
        raise
