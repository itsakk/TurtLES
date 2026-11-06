"""MINI-TurtLES subset definitions (paper, Appendix J).

Subsets cut the dataset along two independent axes:

* geometry tiers ``G4 ⊃ G3 ⊃ G2 ⊃ G1`` keep fewer super-ellipse families ``n``;
* temporal tiers ``T100 ... T400`` keep the first 100 ... 400 snapshots of every
  trajectory, which maps one-to-one onto the 100-frame HDF5 chunks.

``full`` is ``G4-T400`` (13 TB); ``debug`` is the first 100 frames of the
circular-cylinder reference case shipped under ``reference/`` (about 5 GB).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from turtles.data.cases import family_token, parse_case_name

FRAMES_PER_CHUNK = 100
TOTAL_FRAMES = 400
SPLITS: tuple[str, ...] = ("train", "val", "test")
REFERENCE_SPLIT = "reference"
DEBUG_CASE = "Re1000_alpha0_a0p5_n2"

GEOMETRY_TIERS: dict[str, tuple[float, ...]] = {
    "G4": (1.0, 2.0, 5.0, math.inf),
    "G3": (1.0, 2.0, math.inf),
    "G2": (2.0, math.inf),
    "G1": (2.0,),
}
TEMPORAL_TIERS: dict[str, int] = {f"T{k}": k for k in (100, 200, 300, 400)}

_SPEC_RE = re.compile(r"^(G[1-4])-(T[1-4]00)$", re.IGNORECASE)


def chunk_suffix(index: int) -> str:
    """File suffix of the ``index``-th 100-frame chunk (``b0_99``, ``b100_199``, ...)."""
    start = index * FRAMES_PER_CHUNK
    return f"b{start}_{start + FRAMES_PER_CHUNK - 1}"


@dataclass(frozen=True)
class Subset:
    """A standardized subset of the dataset.

    Attributes:
        name: Canonical name (``full``, ``G2-T200``, ``debug``).
        families: Super-ellipse exponents kept.
        num_frames: Snapshots kept per trajectory (multiple of 100).
        debug: Whether this is the single reference-case debug subset.
    """

    name: str
    families: tuple[float, ...]
    num_frames: int
    debug: bool = False

    @property
    def num_chunks(self) -> int:
        """Number of 100-frame HDF5 chunks needed per trajectory."""
        return self.num_frames // FRAMES_PER_CHUNK

    @property
    def chunk_suffixes(self) -> tuple[str, ...]:
        """File suffixes of the chunks needed per trajectory."""
        return tuple(chunk_suffix(i) for i in range(self.num_chunks))

    def accepts_family(self, n: float) -> bool:
        """Whether trajectories with exponent ``n`` belong to the subset."""
        return n in self.families

    def accepts(self, case_name: str) -> bool:
        """Whether the case directory ``case_name`` belongs to the subset."""
        if self.debug:
            return case_name == DEBUG_CASE
        try:
            params = parse_case_name(case_name)
        except ValueError:
            return False
        return self.accepts_family(params.n)

    def split_dir_name(self, split: str) -> str:
        """Directory holding ``split`` (the debug subset lives under ``reference/``)."""
        return REFERENCE_SPLIT if self.debug else split

    def hf_patterns(
        self,
        splits: Iterable[str] = SPLITS,
        *,
        include_vtu: bool = False,
        include_stats: bool = True,
    ) -> list[str]:
        """Glob patterns selecting this subset's files inside the Hugging Face repo."""
        patterns = ["data/stats.yaml"] if include_stats else []
        if self.debug:
            case_dirs = [f"data/{REFERENCE_SPLIT}/{DEBUG_CASE}"]
        else:
            case_dirs = [
                f"data/{split}/*_n{family_token(n)}" for split in splits for n in self.families
            ]
        for case_dir in case_dirs:
            patterns.append(f"{case_dir}/*_coords.h5")
            patterns.extend(f"{case_dir}/*_{suffix}.h5" for suffix in self.chunk_suffixes)
            if include_vtu:
                patterns.append(f"{case_dir}/*_template.vtu")
        return patterns


def parse_subset(spec: str | None) -> Subset:
    """Parse ``full``, ``debug`` or ``G{1-4}-T{100-400}`` (case-insensitive)."""
    text = (spec or "full").strip()
    lowered = text.lower()
    if lowered == "full":
        return Subset("full", GEOMETRY_TIERS["G4"], TOTAL_FRAMES)
    if lowered == "debug":
        return Subset("debug", (2.0,), FRAMES_PER_CHUNK, debug=True)
    match = _SPEC_RE.match(text)
    if match is None:
        raise ValueError(
            f"Unknown subset '{spec}'. Expected 'full', 'debug' or 'G{{1-4}}-T{{100,200,300,400}}'."
        )
    geometry, temporal = match.group(1).upper(), match.group(2).upper()
    return Subset(f"{geometry}-{temporal}", GEOMETRY_TIERS[geometry], TEMPORAL_TIERS[temporal])
