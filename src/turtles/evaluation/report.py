"""Build the paper's result tables and spectral figure from evaluation directories.

* Table 1 (next frame): Rel-L2, VRMSE and the three spectral band errors at step 1.
* Table 2 (stability and cost): VRMSE over the disjoint rollout windows and the
  GPU-seconds per trajectory, with the solver cost as reference.
* Figure: reference and predicted spanwise spectra plus the log discrepancy.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

Row = tuple[str, Mapping[str, float] | None]


def load_metrics(eval_dir: str | Path) -> dict:
    """Read ``metrics.json`` written by :class:`turtles.evaluation.rollout.RolloutEvaluator`."""
    with open(Path(eval_dir) / "metrics.json") as handle:
        return json.load(handle)


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) >= 100 or abs(value) < 1e-2:
        return f"{value:.{digits - 1}e}"
    return f"{value:.{digits}g}"


def _table(
    header: Sequence[str], rows: Iterable[Sequence[str]], fmt: str, caption: str = ""
) -> str:
    rows = list(rows)
    if fmt == "md":
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return ("" if not caption else f"**{caption}**\n\n") + "\n".join(lines) + "\n"
    if fmt == "tex":
        cols = "l" + "c" * (len(header) - 1)
        lines = [
            f"\\begin{{tabular}}{{{cols}}}",
            "\\toprule",
            " & ".join(header) + r" \\",
            "\\midrule",
        ]
        lines += [" & ".join(r) + r" \\" for r in rows]
        lines += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(lines) + "\n"
    raise ValueError(f"unknown table format '{fmt}'")


def next_frame_table(rows: Sequence[Row], band_names: Sequence[str], fmt: str = "md") -> str:
    """Table 1: next-frame field and spectral errors."""
    arrow = " ↓" if fmt == "md" else r" $\downarrow$"
    header = [
        "Model",
        f"Rel-L2{arrow}",
        f"VRMSE{arrow}",
        *(f"{b.capitalize()}{arrow}" for b in band_names),
    ]
    body = []
    for name, m in rows:
        if m is None:
            body.append([name, *(["--"] * (2 + len(band_names)))])
            continue
        body.append([
            name, _fmt(m.get("rel_l2/0_1")), _fmt(m.get("vrmse/0_1")),
            *(_fmt(m.get(f"spec_{b}/0_1")) for b in band_names),
        ])  # fmt: skip
    return _table(header, body, fmt, "Next-frame results (test set)")


def rollout_table(
    rows: Sequence[Row],
    windows: Sequence[tuple[int, int]],
    fmt: str = "md",
    solver_gpu_seconds: float | None = None,
) -> str:
    """Table 2: VRMSE over disjoint rollout windows and GPU-seconds per trajectory."""
    arrow = " ↓" if fmt == "md" else r" $\downarrow$"
    header = ["Model", *(f"{lo}-{hi}" for lo, hi in windows), f"Cost [GPU-s]{arrow}", "Diverged"]
    body = []
    if solver_gpu_seconds:
        body.append(["Solver (PyFR)", *(["--"] * len(windows)), f"{solver_gpu_seconds:.0f}", "--"])
    for name, m in rows:
        if m is None:
            body.append([name, *(["--"] * (len(windows) + 2))])
            continue
        cost = m.get("time/gpu_seconds")
        body.append([
            name, *(_fmt(m.get(f"vrmse/win_{lo}_{hi}")) for lo, hi in windows),
            "--" if cost is None else f"{cost:.0f}",
            "--" if m.get("diverged") is None else f"{100 * m['diverged']:.0f}%",
        ])  # fmt: skip
    return _table(header, body, fmt, "Rollout stability (VRMSE per window) and inference cost")


def mean_spectra(
    eval_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Average the per-trajectory ``spectra.npz`` files of an evaluation directory."""
    refs, preds, logs, k = [], [], [], None
    for npz in sorted(Path(eval_dir).glob("trajectories/*/spectra.npz")):
        with np.load(npz) as data:
            if k is None:
                k = data["wavenumbers"]
            elif data["wavenumbers"].shape != k.shape:
                continue
            refs.append(data["power_ref"])
            preds.append(data["power_pred"])
            logs.append(data["log_discrepancy"])
    if k is None:
        return None
    return k, np.mean(refs, axis=0), np.mean(preds, axis=0), np.mean(logs, axis=0)


def spectral_figure(
    rows: Sequence[tuple[str, Path]], path: str | Path, dpi: int = 150
) -> Path | None:
    """Reference + per-model spanwise spectra (left) and log discrepancy (right)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=dpi)
    drawn = False
    for name, eval_dir in rows:
        spectra = mean_spectra(eval_dir)
        if spectra is None:
            continue
        k, ref, pred, log_disc = spectra
        if not drawn:
            axes[0].loglog(
                k[1:], np.maximum(ref[1:], 1e-30), "k-", lw=2.2, label="Reference", zorder=3
            )
            drawn = True
        axes[0].loglog(k[1:], np.maximum(pred[1:], 1e-30), lw=1.4, label=name)
        axes[1].semilogx(k[1:], log_disc[1:], lw=1.4, label=name)
    if not drawn:
        plt.close(fig)
        return None
    axes[0].set_ylabel(r"$E(k_z)$")
    axes[0].set_title("(a) Energy spectra")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].axhline(0.0, color="k", lw=0.8, alpha=0.6)
    axes[1].set_ylabel(r"$\overline{\log E_{pred} - \log E_{ref}}$")
    axes[1].set_title("(b) Spectral error")
    for ax in axes:
        ax.set_xlabel(r"spanwise wavenumber $k_z$")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_report(
    evals: Mapping[str, Path],
    out_dir: str | Path,
    *,
    formats: Sequence[str] = ("md", "tex"),
    figure: bool = True,
) -> list[Path]:
    """Write the tables (and figure) for the given ``{display name: eval dir}`` mapping."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    band_names: list[str] = []
    windows: list[tuple[int, int]] = []
    solver: float | None = None
    for name, directory in evals.items():
        try:
            report = load_metrics(directory)
        except FileNotFoundError:
            rows.append((name, None))
            continue
        aggregate = report["aggregate"]
        settings = report.get("settings", {})
        band_names = band_names or list(settings.get("spectral_bands", {}))
        windows = windows or [tuple(w) for w in settings.get("disjoint_windows", [])]
        solver = solver or settings.get("solver_gpu_seconds")
        rows.append((name, aggregate))

    written: list[Path] = []
    for fmt in formats:
        for stem, text in (
            ("next_frame", next_frame_table(rows, band_names, fmt)),
            ("rollout", rollout_table(rows, windows, fmt, solver)),
        ):
            path = out_dir / f"{stem}.{fmt}"
            path.write_text(text)
            written.append(path)
    if figure:
        fig_path = spectral_figure([(n, d) for n, d in evals.items()], out_dir / "spectra.pdf")
        if fig_path is not None:
            written.append(fig_path)
    return written
