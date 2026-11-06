"""Figures written next to the rollout metrics (matplotlib, Agg backend)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_snapshots(
    reference: np.ndarray,
    prediction: np.ndarray,
    steps: Sequence[int],
    *,
    title: str,
    path: str | Path,
    extent: tuple[float, float, float, float] | None = None,
    dpi: int = 130,
) -> None:
    """Reference / prediction / |error| rows for a few rollout steps (``(S, H, W)`` inputs)."""
    error = np.abs(prediction - reference)
    finite = np.isfinite(reference) & np.isfinite(prediction)
    vmin = float(np.min(reference[finite])) if finite.any() else 0.0
    vmax = float(np.max(reference[finite])) if finite.any() else 1.0
    emax = float(np.nanmax(np.where(np.isfinite(error), error, np.nan))) if finite.any() else 1.0
    ncols = len(steps)
    fig, axes = plt.subplots(3, ncols, figsize=(2.4 * ncols, 6.4), dpi=dpi, squeeze=False)
    fig.suptitle(title)
    for j, step in enumerate(steps):
        axes[0, j].imshow(
            reference[j], cmap="viridis", vmin=vmin, vmax=vmax, origin="lower", extent=extent
        )
        axes[1, j].imshow(
            prediction[j], cmap="viridis", vmin=vmin, vmax=vmax, origin="lower", extent=extent
        )
        axes[2, j].imshow(
            error[j], cmap="inferno", vmin=0.0, vmax=emax, origin="lower", extent=extent
        )
        axes[0, j].set_title(f"step {step}")
        for row in range(3):
            axes[row, j].set_xticks([])
            axes[row, j].set_yticks([])
    for row, label in enumerate(("reference", "prediction", "|error|")):
        axes[row, 0].set_ylabel(label)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_error_curves(
    rel_l2: np.ndarray,
    vrmse: np.ndarray,
    channel_names: Sequence[str],
    *,
    path: str | Path,
    dpi: int = 130,
) -> None:
    """Rel-L2 and VRMSE versus rollout step, per channel and channel mean (``(T, C)`` inputs)."""
    steps = np.arange(1, rel_l2.shape[0] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=dpi, sharex=True)
    for ax, values, label in zip(axes, (rel_l2, vrmse), ("Rel-L2", "VRMSE"), strict=True):
        for c, name in enumerate(channel_names):
            ax.plot(steps, values[:, c], lw=1.0, alpha=0.8, label=name)
        ax.plot(steps, values.mean(axis=1), "k-", lw=1.8, label="mean")
        ax.set_yscale("log")
        ax.set_xlabel("rollout step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_spectra(
    wavenumbers: np.ndarray,
    power_ref: np.ndarray,
    power_pred: np.ndarray,
    log_discrepancy: np.ndarray,
    *,
    label: str,
    path: str | Path,
    dpi: int = 130,
) -> None:
    """Time-averaged spanwise spectra (reference vs prediction) and the log discrepancy."""
    k = wavenumbers[1:]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=dpi)
    axes[0].loglog(k, np.maximum(power_ref[1:], 1e-30), "k-", lw=2.0, label="reference")
    axes[0].loglog(k, np.maximum(power_pred[1:], 1e-30), "-", lw=1.4, label=label)
    axes[0].set_xlabel(r"spanwise wavenumber $k_z$")
    axes[0].set_ylabel(r"$E(k_z)$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].semilogx(k, log_discrepancy[1:], lw=1.4)
    axes[1].axhline(0.0, color="k", lw=0.8, alpha=0.6)
    axes[1].set_xlabel(r"spanwise wavenumber $k_z$")
    axes[1].set_ylabel(r"$\overline{\log E_{pred} - \log E_{ref}}$")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
