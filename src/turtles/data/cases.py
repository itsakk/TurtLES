"""Case naming: ``Re{Re}_alpha{alpha}_a{a}_n{n}`` <-> physical parameters."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

PARAM_NAMES: tuple[str, ...] = ("Re", "alpha", "a", "n")

_NAME_RE = re.compile(r"^Re(?P<Re>\d+)_alpha(?P<alpha>-?\d+)_a(?P<a>\d+(?:p\d+)?)_n(?P<n>inf|\d+)$")


def family_token(n: float) -> str:
    """Return the super-ellipse exponent as it appears in case names (``2``, ``inf``...)."""
    return "inf" if math.isinf(n) else str(int(n))


def format_case_name(Re: float, alpha: float, a: float, n: float) -> str:
    """Build the canonical case directory name used by the generation pipeline.

    ``a`` is rounded to three decimals and written with ``p`` as decimal separator
    (``0.5 -> 0p5``, ``1.302 -> 1p302``), matching ``setup_simulation`` in the pipeline.
    """
    a_token = str(round(float(a), 3)).replace(".", "p")
    return f"Re{round(Re)}_alpha{round(alpha)}_a{a_token}_n{family_token(n)}"


@dataclass(frozen=True)
class CaseParams:
    """Physical parameters of one simulation case.

    Attributes:
        Re: Reynolds number based on the obstacle thickness ``D = 1``.
        alpha: Angle of attack in degrees.
        a: Streamwise semi-axis of the super-ellipse (``b = 1/2`` is fixed).
        n: Super-ellipse exponent (``1`` diamond, ``2`` ellipse, ``5`` rounded
            rectangle, ``inf`` rectangle).
    """

    Re: float
    alpha: float
    a: float
    n: float

    @property
    def name(self) -> str:
        """Canonical case name."""
        return format_case_name(self.Re, self.alpha, self.a, self.n)

    def as_dict(self) -> dict[str, float]:
        """Parameters keyed by name, in ``PARAM_NAMES`` order."""
        return {"Re": self.Re, "alpha": self.alpha, "a": self.a, "n": self.n}


def parse_case_name(name: str) -> CaseParams:
    """Parse a case directory name into :class:`CaseParams`.

    Raises:
        ValueError: if ``name`` does not follow ``Re{Re}_alpha{alpha}_a{a}_n{n}``.
    """
    match = _NAME_RE.match(name)
    if match is None:
        raise ValueError(f"'{name}' is not a TurtLES case name (Re*_alpha*_a*_n*)")
    n_token = match.group("n")
    return CaseParams(
        Re=float(match.group("Re")),
        alpha=float(match.group("alpha")),
        a=float(match.group("a").replace("p", ".")),
        n=math.inf if n_token == "inf" else float(n_token),
    )
