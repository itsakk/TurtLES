"""Stationarity criterion of the wake probes (paper, Appendix H; Boudet et al., 2015).

The transient run records ``u`` and ``v`` at three centerline rakes behind the
object (64 spanwise probes each). For a signal cut into four equal time quarters,
``chi`` is the largest relative change of the spanwise/time mean and standard
deviation between quarters 2-4, relative to the last quarter. Once ``chi`` settles
(below ~10 %), the transient has left the wake and the trajectory can be recorded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "the convergence check needs pandas: pip install 'turtles[generation]'"
    ) from exc


def compute_chi(means: np.ndarray, stds: np.ndarray, *, mean_is_zero: bool) -> float:
    """``chi`` in percent from per-quarter means and standard deviations (4 entries each)."""
    means_q = np.asarray(means[1:], dtype=float)
    stds_q = np.asarray(stds[1:], dtype=float)
    mean_ref = float(means[-1])
    std_ref = float(stds[-1])
    if mean_is_zero:  # v has zero mean: only the fluctuation level is meaningful
        means_q = np.zeros_like(means_q)
        mean_ref = 1.0
    mean_diff = np.abs(np.subtract.outer(means_q, means_q)) / abs(mean_ref + 1e-9)
    std_diff = np.abs(np.subtract.outer(stds_q, stds_q)) / (std_ref + 1e-9)
    return float(max(mean_diff.max(), std_diff.max()) * 100.0)


def _quarter_stats(df: pd.DataFrame, t_limit: float) -> pd.DataFrame:
    """Per (quarter, x) mean/std of u and v after resampling every probe to a common dt."""
    sub = df[df["t"] <= t_limit].copy()
    sub["q"] = np.clip(np.floor(sub["t"] / (t_limit / 4 + 1e-9)), 0, 3).astype(int)
    records = []
    for q, block in sub.groupby("q"):
        times = np.sort(block["t"].unique())
        dt = float(np.diff(times).max()) if len(times) > 1 else 0.0
        for (x, _y, _z), probe in block.groupby(["x", "y", "z"]):
            probe = probe.sort_values("t")
            if dt > 0:
                t_new = np.arange(probe["t"].iloc[0], probe["t"].iloc[-1] + dt / 2, dt)
                u = np.interp(t_new, probe["t"], probe["u"])
                v = np.interp(t_new, probe["t"], probe["v"])
            else:
                u, v = probe["u"].to_numpy(), probe["v"].to_numpy()
            records.append(pd.DataFrame({"q": q, "x": round(float(x), 4), "u": u, "v": v}))
    stacked = pd.concat(records, ignore_index=True)
    return (
        stacked.groupby(["q", "x"])
        .agg(u_mean=("u", "mean"), u_std=("u", "std"), v_mean=("v", "mean"), v_std=("v", "std"))
        .reset_index()
    )


def convergence_history(
    probe_csv: str | Path, *, num_times: int = 50, t_start: float = 50.0
) -> pd.DataFrame:
    """``chi(u)`` and ``chi(v)`` per rake as the analysed time span grows.

    Returns a frame with columns ``time, x, chi_u, chi_v``.
    """
    df = pd.read_csv(probe_csv, usecols=["t", "x", "y", "z", "u", "v"])
    t_max = float(df["t"].max())
    rows = []
    for t_limit in np.linspace(t_start, t_max, num_times):
        stats = _quarter_stats(df, t_limit)
        for x, block in stats.groupby("x"):
            block = block.sort_values("q")
            if len(block) < 4:
                continue
            rows.append(
                {
                    "time": t_limit,
                    "x": x,
                    "chi_u": compute_chi(
                        block["u_mean"].to_numpy(), block["u_std"].to_numpy(), mean_is_zero=False
                    ),
                    "chi_v": compute_chi(
                        block["v_mean"].to_numpy(), block["v_std"].to_numpy(), mean_is_zero=True
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_convergence(
    history: pd.DataFrame, path: str | Path, *, velocity: float = 0.2366431913
) -> Path:
    """Log-scale ``chi`` versus convective time for each rake."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for x, block in history.groupby("x"):
        axes[0].plot(block["time"] * velocity, block["chi_u"], lw=2, label=f"x = {x:.2f}")
        axes[1].plot(block["time"] * velocity, block["chi_v"], lw=2, label=f"x = {x:.2f}")
    for ax, label in zip(axes, (r"$\chi(u)$ [%]", r"$\chi(v)$ [%]"), strict=True):
        ax.set_yscale("log")
        ax.set_xlabel("convective time")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.4)
        ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
