"""Quasi-random design of the parameter space (paper, Section 3).

Parameters are drawn from a scrambled Sobol' sequence on the unit hypercube and
mapped to Re ∈ [1000, 3900] (integer), alpha ∈ [0, 45] degrees (integer), a
log-uniform in [1/6, 3/2] (3 decimals) and n ∈ {1, 2, 5, inf} (uniform). Cases are
assigned to train / val / test with an 8/1/1 pattern on the sample index, so the
design can be extended later without re-shuffling earlier cases.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from turtles.data.cases import CaseParams

CSV_COLUMNS = ("index", "split", "Re", "alpha", "a", "n", "name")


@dataclass(frozen=True)
class ParameterSpace:
    """Sampling ranges. Integer parameters include ±0.5 so the bounds are equiprobable."""

    re_range: tuple[float, float] = (999.5, 3900.5)
    alpha_range: tuple[float, float] = (-0.5, 45.5)
    log10_a_range: tuple[float, float] = (math.log10(1.0 / 6.0), math.log10(1.5))
    n_choices: tuple[float, ...] = (1.0, 2.0, 5.0, math.inf)

    def transform(self, unit: np.ndarray) -> list[CaseParams]:
        """Map ``(m, 4)`` unit-cube samples to physical parameters."""
        re = np.round(self.re_range[0] + unit[:, 0] * (self.re_range[1] - self.re_range[0]))
        alpha = np.round(
            self.alpha_range[0] + unit[:, 1] * (self.alpha_range[1] - self.alpha_range[0])
        )
        log_a = self.log10_a_range[0] + unit[:, 2] * (self.log10_a_range[1] - self.log10_a_range[0])
        re = np.clip(re, math.ceil(self.re_range[0]), math.floor(self.re_range[1]))
        alpha = np.clip(alpha, math.ceil(self.alpha_range[0]), math.floor(self.alpha_range[1]))
        a = np.round(10.0**log_a, 3)
        n_idx = np.clip(
            np.floor(unit[:, 3] * len(self.n_choices)).astype(int), 0, len(self.n_choices) - 1
        )
        return [
            CaseParams(
                Re=float(re[i]), alpha=float(alpha[i]), a=float(a[i]), n=self.n_choices[n_idx[i]]
            )
            for i in range(unit.shape[0])
        ]


def assign_split(index: int) -> str:
    """8/1/1 train/val/test assignment from the sample index."""
    tag = index % 10
    return "train" if tag < 8 else "val" if tag == 8 else "test"


@dataclass
class DesignRow:
    """One sampled case with its index and split."""

    index: int
    split: str
    params: CaseParams

    @property
    def name(self) -> str:
        """Case directory name."""
        return self.params.name


@dataclass
class SobolDesign:
    """Scrambled Sobol' design of ``m`` cases (``m`` should be a power of two)."""

    space: ParameterSpace = field(default_factory=ParameterSpace)
    seed: int = 42

    def generate(self, m: int) -> list[DesignRow]:
        """Draw ``m`` cases."""
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=4, scramble=True, seed=self.seed)
        unit = sampler.random(m)
        params = self.space.transform(unit)
        return [DesignRow(i, assign_split(i), p) for i, p in enumerate(params)]


def write_design(rows: Sequence[DesignRow], path: str | Path) -> Path:
    """Write the design as CSV (``index, split, Re, alpha, a, n, name``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            p = row.params
            n_text = "inf" if math.isinf(p.n) else str(int(p.n))
            writer.writerow(
                [row.index, row.split, int(p.Re), int(p.alpha), f"{p.a:.3f}", n_text, row.name]
            )
    return path


def read_design(path: str | Path) -> list[DesignRow]:
    """Read a design CSV written by :func:`write_design`."""
    rows: list[DesignRow] = []
    with open(path, newline="") as handle:
        for record in csv.DictReader(handle):
            n_text = record["n"].strip().lower()
            params = CaseParams(
                Re=float(record["Re"]),
                alpha=float(record["alpha"]),
                a=float(record["a"]),
                n=math.inf if n_text == "inf" else float(n_text),
            )
            rows.append(DesignRow(int(record["index"]), record["split"], params))
    return rows
